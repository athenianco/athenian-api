from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
import logging
import pickle
from typing import Any, Callable, Collection, Iterable, KeysView, Mapping, Optional

import aiomcache
import morcilla
import numpy as np
from numpy import typing as npt
import pandas as pd
import sentry_sdk
from sqlalchemy import (
    and_,
    delete,
    exists,
    false,
    func,
    join,
    not_,
    or_,
    select,
    text,
    true,
    union_all,
    update,
)
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement, Select

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.db import Database, dialect_specific_insert, is_postgresql
from athenian.api.internal.logical_accelerated import mark_logical_repos_in_list
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.filters import LabelFilter
from athenian.api.internal.miners.github.commit import (
    BRANCH_FETCH_COMMITS_COLUMNS,
    fetch_precomputed_commit_history_dags,
    fetch_repository_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.internal.miners.github.precomputed_prs.utils import (
    append_activity_days_filter,
    build_labels_filters,
    collect_activity_days,
    labels_are_compatible,
    triage_by_release_match,
)
from athenian.api.internal.miners.github.release_load import (
    group_repos_by_release_match,
    match_groups_to_sql,
)
from athenian.api.internal.miners.github.released_pr import matched_by_column, new_released_prs_df
from athenian.api.internal.miners.participation import PRParticipants, PRParticipationKind
from athenian.api.internal.miners.types import (
    MinedPullRequest,
    PullRequestFacts,
    PullRequestFactsMap,
    PullRequestID,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import (
    NodeCommit,
    NodePullRequest,
    PullRequest,
    PullRequestLabel,
    Release,
)
from athenian.api.models.precomputed.models import (
    GitHubDonePullRequestFacts,
    GitHubPullRequestDeployment,
)
from athenian.api.tracing import sentry_span


class DonePRFactsLoader:
    """Loader for done PRs facts."""

    @classmethod
    @sentry_span
    async def load_precomputed_done_candidates(
        cls,
        time_from: datetime,
        time_to: datetime,
        repos: Collection[str],
        default_branches: dict[str, str],
        release_settings: ReleaseSettings,
        account: int,
        pdb: morcilla.Database,
    ) -> tuple[npt.NDArray[int], dict[str, list[int]]]:
        """
        Load the set of done PR identifiers and specifically ambiguous PR node IDs.

        We find all the done PRs for a given time frame, repositories, and release match settings.

        Note: we don't include the deployed PRs!

        :return: 1. Done PR node IDs, unique sorted. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        ghprt = GitHubDonePullRequestFacts
        selected = [ghprt.pr_node_id, ghprt.repository_full_name, ghprt.release_match]
        filters = cls._create_common_filters(time_from, time_to, repos, account)
        with sentry_sdk.start_span(op="load_precomputed_done_candidates/fetch"):
            rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        pr_node_id_col = 0  # performance: faster than ghprt.pr_node_id.name
        repository_full_name_col = 1  # performance: faster than ghprt.repository_full_name.name
        release_match_col = 2  # performance: faster than ghprt.release_match.name
        for row in rows:
            dump = triage_by_release_match(
                row[repository_full_name_col],
                row[release_match_col],
                release_settings,
                default_branches,
                result,
                ambiguous,
            )
            if dump is None:
                continue
            dump[(row[pr_node_id_col], row[repository_full_name_col])] = row
        result, ambiguous = cls._post_process_ambiguous_done_prs(result, ambiguous)
        return (
            np.unique(np.fromiter((node_id for node_id, _ in result), int, len(result))),
            ambiguous,
        )

    @classmethod
    @sentry_span
    async def load_precomputed_done_facts_filters(
        cls,
        time_from: datetime,
        time_to: datetime,
        repos: Collection[str],
        participants: PRParticipants,
        labels: LabelFilter,
        default_branches: dict[str, str],
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
    ) -> tuple[PullRequestFactsMap, dict[str, list[int]]]:
        """
        Fetch precomputed done PR facts.

        :return: 1. Map from PR node IDs to repo names and facts. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        ghdprf = GitHubDonePullRequestFacts
        assert time_from is not None
        assert time_to is not None
        result, ambiguous = await cls._load_precomputed_done_filters(
            [ghdprf.data, ghdprf.author, ghdprf.merger, ghdprf.releaser],
            time_from,
            time_to,
            repos,
            participants,
            labels,
            default_branches,
            exclude_inactive,
            release_settings,
            prefixer,
            account,
            pdb,
        )
        user_node_to_login_get = prefixer.user_node_to_login.get
        data_name = ghdprf.data.name
        pr_node_id_name = ghdprf.pr_node_id.name
        repository_full_name_name = ghdprf.repository_full_name.name
        author_name = ghdprf.author.name
        merger_name = ghdprf.merger.name
        releaser_name = ghdprf.releaser.name
        for key, row in result.items():
            result[key] = PullRequestFacts(
                data=row[data_name],
                node_id=row[pr_node_id_name],
                repository_full_name=row[repository_full_name_name],
                author=user_node_to_login_get(row[author_name], ""),
                merger=user_node_to_login_get(row[merger_name], ""),
                releaser=user_node_to_login_get(row[releaser_name], ""),
            )
            # optimization over cls._done_pr_facts_from_row(row, user_node_to_login_get)
        return result, ambiguous

    @classmethod
    async def load_precomputed_done_facts_all(
        cls,
        repos: Collection[str],
        default_branches: dict[str, str],
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
        extra: Iterable[InstrumentedAttribute] = (),
    ) -> tuple[PullRequestFactsMap, dict[str, Mapping[str, Any]]]:
        """
        Fetch all the precomputed done PR facts we have.

        We don't set the repository, the author, and the merger!

        :param extra: Additional columns to fetch.

        :return: 1. Map from PR node IDs to repo names and facts. \
                 2. Map from PR node IDs to raw returned rows.
        """
        ghdprf = GitHubDonePullRequestFacts
        result, _ = await cls._load_precomputed_done_filters(
            [ghdprf.data, ghdprf.release_match, ghdprf.releaser, *extra],
            None,
            None,
            repos,
            {},
            LabelFilter.empty(),
            default_branches,
            False,
            release_settings,
            prefixer,
            account,
            pdb,
        )
        raw = {}
        user_node_to_login_get = prefixer.user_node_to_login.get
        data_name = ghdprf.data.name
        releaser_name = ghdprf.releaser.name
        for (node_id, repo), row in result.items():
            result[(node_id, repo)] = PullRequestFacts(
                data=row[data_name],
                node_id=node_id,
                releaser=user_node_to_login_get(row[releaser_name]),
            )
            raw[node_id] = row
        return result, raw

    @classmethod
    @sentry_span
    async def load_precomputed_done_timestamp_filters(
        cls,
        time_from: datetime,
        time_to: datetime,
        repos: Collection[str],
        participants: PRParticipants,
        labels: LabelFilter,
        default_branches: dict[str, str],
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
    ) -> tuple[dict[PullRequestID, datetime], dict[str, list[int]]]:
        """
        Fetch precomputed done PR "pr_done_at" timestamps.

        :return: 1. map from PR IDs to their release timestamps. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        prs, ambiguous = await cls._load_precomputed_done_filters(
            [GitHubDonePullRequestFacts.pr_done_at],
            time_from,
            time_to,
            repos,
            participants,
            labels,
            default_branches,
            exclude_inactive,
            release_settings,
            prefixer,
            account,
            pdb,
        )
        result = {key: row[GitHubDonePullRequestFacts.pr_done_at.name] for key, row in prs.items()}
        if pdb.url.dialect == "sqlite":
            for key, dt in result.items():
                result[key] = dt.replace(tzinfo=timezone.utc)
        return result, ambiguous

    @classmethod
    @sentry_span
    async def load_precomputed_done_facts_reponums(
        cls,
        repos: dict[str, set[int]],
        default_branches: dict[str, str],
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
    ) -> tuple[PullRequestFactsMap, dict[str, list[int]]]:
        """
        Load PullRequestFacts belonging to released or rejected PRs from the precomputed DB.

        repo + numbers version.

        :return: 1. Map PR node ID -> repository name & specified column value. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        ghprt = GitHubDonePullRequestFacts
        selected = [
            ghprt.pr_node_id,
            ghprt.repository_full_name,
            ghprt.release_match,
            ghprt.data,
            ghprt.author,
            ghprt.merger,
            ghprt.releaser,
        ]
        format_version_filter = (
            ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg
        )
        if pdb.url.dialect == "sqlite":
            filters = [
                format_version_filter,
                or_(
                    *[
                        and_(
                            ghprt.acc_id == account,
                            ghprt.repository_full_name == repo,
                            ghprt.number.in_(numbers),
                        )
                        for repo, numbers in repos.items()
                    ],
                ),
            ]
            query = select(selected).where(and_(*filters))
        else:
            match_groups, _ = group_repos_by_release_match(
                repos, default_branches, release_settings,
            )
            match_groups[ReleaseMatch.rejected] = match_groups[ReleaseMatch.force_push_drop] = {
                "": repos,
            }
            or_items, or_repos = match_groups_to_sql(match_groups, ghprt, False)
            query = union_all(
                *(
                    select(selected).where(
                        and_(
                            ghprt.acc_id == account,
                            item,
                            format_version_filter,
                            ghprt.repository_full_name.in_(item_repos),
                            or_(
                                *(
                                    and_(
                                        ghprt.repository_full_name == repo,
                                        ghprt.number.in_(repos[repo]),
                                    )
                                    for repo in item_repos
                                ),
                            ),
                        ),
                    )
                    for item, item_repos in zip(or_items, or_repos)
                ),
            )

        with sentry_sdk.start_span(op="load_precomputed_done_facts_reponums/fetch"):
            rows = await pdb.fetch_all(query)
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        user_node_to_login_get = prefixer.user_node_to_login.get
        data_name = ghprt.data.name
        pr_node_id_name = ghprt.pr_node_id.name
        repository_full_name_name = ghprt.repository_full_name.name
        release_match_name = ghprt.release_match.name
        author_name = ghprt.author.name
        merger_name = ghprt.merger.name
        releaser_name = ghprt.releaser.name
        _done_pr_facts_from_row = cls._done_pr_facts_from_row
        for row in rows:
            repo = row[repository_full_name_name]
            dump = triage_by_release_match(
                repo,
                row[release_match_name],
                release_settings,
                default_branches,
                result,
                ambiguous,
            )
            if dump is None:
                continue
            dump[(row[pr_node_id_name], repo)] = _done_pr_facts_from_row(
                row,
                user_node_to_login_get,
                data_name,
                pr_node_id_name,
                repository_full_name_name,
                author_name,
                merger_name,
                releaser_name,
            )
        return cls._post_process_ambiguous_done_prs(result, ambiguous)

    @classmethod
    @sentry_span
    async def load_precomputed_done_facts_ids(
        cls,
        node_ids: Iterable[int],
        default_branches: dict[str, str],
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
        panic_on_missing_repositories: bool = True,
    ) -> tuple[PullRequestFactsMap, dict[str, list[int]]]:
        """
        Load PullRequestFacts belonging to released or rejected PRs from the precomputed DB.

        node ID version.

        :param panic_on_missing_repositories: Whether to assert that `release_settings` contain \
          all the loaded PR repositories. If `False`, we log warnings and discard the offending \
          PRs.

        :return: 1. Map (PR node ID, repository name) -> facts. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        log = logging.getLogger(f"{metadata.__package__}.load_precomputed_done_facts_ids")
        ghprt = GitHubDonePullRequestFacts
        selected = [
            ghprt.pr_node_id,
            ghprt.repository_full_name,
            ghprt.release_match,
            ghprt.data,
            ghprt.author,
            ghprt.merger,
            ghprt.releaser,
        ]
        filters = [
            ghprt.acc_id == account,
            ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
            ghprt.pr_node_id.in_(node_ids),
        ]
        query = select(selected).where(and_(*filters))
        with sentry_sdk.start_span(op="load_precomputed_done_facts_ids/fetch"):
            rows = await pdb.fetch_all(query)
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        user_node_to_login_get = prefixer.user_node_to_login.get
        data_name = ghprt.data.name
        pr_node_id_name = ghprt.pr_node_id.name
        repository_full_name_name = ghprt.repository_full_name.name
        release_match_name = ghprt.release_match.name
        author_name = ghprt.author.name
        merger_name = ghprt.merger.name
        releaser_name = ghprt.releaser.name
        _done_pr_facts_from_row = cls._done_pr_facts_from_row
        for row in rows:
            repo = row[repository_full_name_name]
            if not panic_on_missing_repositories and repo not in release_settings.native:
                log.warning(
                    "Discarding PR %s because repository %s is missing",
                    row[pr_node_id_name],
                    repo,
                )
                continue
            dump = triage_by_release_match(
                repo,
                row[release_match_name],
                release_settings,
                default_branches,
                result,
                ambiguous,
            )
            if dump is None:
                continue
            dump[(row[pr_node_id_name], repo)] = _done_pr_facts_from_row(
                row,
                user_node_to_login_get,
                data_name,
                pr_node_id_name,
                repository_full_name_name,
                author_name,
                merger_name,
                releaser_name,
            )
        return cls._post_process_ambiguous_done_prs(result, ambiguous)

    @classmethod
    @sentry_span
    @cached(
        exptime=60 * 60,  # 1 hour
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda prs, time_to, default_branches, release_settings, **_: (
            ",".join(map(str, sorted(prs.index.values))),
            time_to,
            sorted(default_branches.items()),
            release_settings,
        ),
        refresh_on_access=True,
    )
    async def load_precomputed_pr_releases(
        cls,
        prs: pd.DataFrame,
        time_to: datetime,
        matched_bys: dict[str, ReleaseMatch],
        default_branches: dict[str, str],
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
        cache: Optional[aiomcache.Client],
    ) -> pd.DataFrame:
        """
        Load the releases mentioned in the specified PRs.

        Each PR is represented by a node_id, a repository name, and a required release match.
        """
        log = logging.getLogger("%s.load_precomputed_pr_releases" % metadata.__package__)
        assert isinstance(time_to, datetime)
        assert time_to.tzinfo is not None
        assert prs.index.nlevels == 2
        ghprt = GitHubDonePullRequestFacts
        pr_node_ids = prs.index.get_level_values(0).values
        query = select(
            ghprt.pr_node_id,
            ghprt.pr_done_at,
            ghprt.releaser,
            ghprt.release_url,
            ghprt.release_node_id,
            ghprt.repository_full_name,
            ghprt.release_match,
        ).where(
            ghprt.acc_id == account,
            ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
            ghprt.releaser.isnot(None),
            ghprt.pr_done_at < time_to,
            ghprt.repository_full_name.in_(prs.index.get_level_values(1).unique()),
            ghprt.pr_node_id.in_(pr_node_ids)
            if len(pr_node_ids) < 500
            else ghprt.pr_node_id.in_any_values(pr_node_ids),
        )
        with sentry_sdk.start_span(op="load_precomputed_pr_releases/fetch"):
            rows = await pdb.fetch_all(query)
        records = []
        utc = timezone.utc
        force_push_dropped = set()
        user_node_to_login_get = prefixer.user_node_to_login.get
        alternative_matched = []
        repository_full_name_col = ghprt.repository_full_name.name
        pr_node_id_col = ghprt.pr_node_id.name
        release_match_col = ghprt.release_match.name
        pr_done_at_col = ghprt.pr_done_at.name
        releaser_col = ghprt.releaser.name
        release_url_col = ghprt.release_url.name
        release_node_id_col = ghprt.release_node_id.name
        force_push_drop_name = ReleaseMatch.force_push_drop.name
        force_push_drop_match = ReleaseMatch.force_push_drop
        event_name = ReleaseMatch.event.name
        event_match = ReleaseMatch.event
        for row in rows:
            repo = row[repository_full_name_col]
            node_id = row[pr_node_id_col]
            release_match = row[release_match_col]
            author_node_id = row[releaser_col]
            if release_match == force_push_drop_name or release_match == event_name:
                if release_match == force_push_drop_name:
                    if node_id in force_push_dropped:
                        continue
                    force_push_dropped.add(node_id)
                    release_match = force_push_drop_match
                else:
                    release_match = event_match
                records.append(
                    (
                        node_id,
                        row[pr_done_at_col].replace(tzinfo=utc),
                        user_node_to_login_get(author_node_id),
                        author_node_id,
                        row[release_url_col],
                        row[release_node_id_col],
                        row[repository_full_name_col],
                        release_match,
                    ),
                )
                continue
            try:
                release_setting = release_settings.native[repo].with_match(matched_bys[repo])
            except KeyError:
                # pdb thinks this PR was released but our current release matching settings
                # disagree
                alternative_matched.append((node_id, repo, release_match))
                continue
            if not release_setting.compatible_with_db(
                release_match, default_branches[drop_logical_repo(repo)],
            ):
                continue
            records.append(
                (
                    node_id,
                    row[pr_done_at_col].replace(tzinfo=utc),
                    user_node_to_login_get(author_node_id),
                    author_node_id,
                    row[release_url_col],
                    row[release_node_id_col],
                    row[repository_full_name_col],
                    release_setting.match,
                ),
            )
        if alternative_matched:
            log.warning(
                "Alternative release matching detected in %d PRs: %s",
                len(alternative_matched),
                alternative_matched,
            )
        return new_released_prs_df(records)

    @classmethod
    @sentry_span
    async def _load_precomputed_done_filters(
        cls,
        columns: list[InstrumentedAttribute],
        time_from: Optional[datetime],
        time_to: Optional[datetime],
        repos: Collection[str],
        participants: PRParticipants,
        labels: LabelFilter,
        default_branches: dict[str, str],
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
    ) -> tuple[dict[PullRequestID, Mapping[str, Any]], dict[str, list[int]]]:
        """
        Load some data belonging to released or rejected PRs from the precomputed DB.

        Query version. JIRA must be filtered separately.
        :return: 1. Map (PR node ID, repository name) -> facts. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        if not isinstance(repos, (set, KeysView)):
            repos = set(repos)
        postgres = pdb.url.dialect == "postgresql"
        ghprt = GitHubDonePullRequestFacts
        selected = {
            ghprt.pr_node_id,
            ghprt.repository_full_name,
            ghprt.release_match,
        }.union(columns)
        match_groups, _ = group_repos_by_release_match(repos, default_branches, release_settings)
        match_groups[ReleaseMatch.rejected] = match_groups[ReleaseMatch.force_push_drop] = {
            "": list(repos),
        }

        def or_items():
            return match_groups_to_sql(match_groups, ghprt, False)[0]

        queries_undeployed, date_range = await cls._compose_query_filters_undeployed(
            selected,
            or_items,
            time_from,
            time_to,
            participants,
            labels,
            exclude_inactive,
            prefixer,
            account,
            postgres,
        )
        queries_deployed = await cls._compose_query_filters_deployed(
            selected,
            or_items,
            time_from,
            time_to,
            participants,
            labels,
            exclude_inactive,
            prefixer,
            account,
            postgres,
        )
        queries = queries_undeployed + queries_deployed
        batch_size = 10
        batches = [
            pdb.fetch_all(union_all(*queries[i : i + batch_size]))
            for i in range(0, len(queries), batch_size)
        ]
        with sentry_sdk.start_span(op="_load_precomputed_done_filters/fetch"):
            rows = list(chain.from_iterable(await gather(*batches)))
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        if labels and not postgres:
            include_singles, include_multiples = LabelFilter.split(labels.include)
            include_singles = set(include_singles)
            include_multiples = [set(m) for m in include_multiples]
        if len(participants) > 0 and not postgres:
            user_node_to_login_get = prefixer.user_node_to_login.get
        repository_full_name_name = ghprt.repository_full_name.name
        logical_repos = [row[repository_full_name_name] for row in rows]
        blocked_physical = set()
        release_match_name = ghprt.release_match.name
        pr_node_id_name = ghprt.pr_node_id.name
        # make logical repos come before physical
        for i in np.argpartition(*mark_logical_repos_in_list(logical_repos))[::-1]:
            repo, row = logical_repos[i], rows[i]
            rm, pr_node_id = row[release_match_name], row[pr_node_id_name]
            if (key := (pr_node_id, repo)) in blocked_physical:
                continue
            dump = triage_by_release_match(
                repo, rm, release_settings, default_branches, result, ambiguous,
            )
            if dump is None:
                continue
            if not postgres:
                if len(participants) > 0 and not await cls._check_participants(
                    row, participants, user_node_to_login_get,
                ):
                    continue
                if labels and not labels_are_compatible(
                    include_singles, include_multiples, labels.exclude, row[ghprt.labels.name],
                ):
                    continue
                if exclude_inactive and not row["deployed"]:
                    activity_days = {
                        datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        for d in row[ghprt.activity_days.name]
                    }
                    if not activity_days.intersection(date_range):
                        continue
            if (physical_repo := drop_logical_repo(repo)) != repo:
                blocked_physical.add((pr_node_id, physical_repo))
            dump[key] = row

        return cls._post_process_ambiguous_done_prs(result, ambiguous)

    @classmethod
    async def _compose_query_filters_undeployed(
        cls,
        selected: set[InstrumentedAttribute],
        or_items: Callable[[], list[ClauseElement]],
        time_from: Optional[datetime],
        time_to: Optional[datetime],
        participants: PRParticipants,
        labels: LabelFilter,
        exclude_inactive: bool,
        prefixer: Prefixer,
        account: int,
        postgres: bool,
    ) -> tuple[list[Select], set[datetime]]:
        ghprt = GitHubDonePullRequestFacts
        filters = cls._create_common_filters(time_from, time_to, None, account)
        selected = selected.copy()
        selected.add(false().label("deployed"))
        if len(participants) > 0:
            await cls._build_participants_filters(
                participants, filters, selected, postgres, prefixer,
            )
        if labels:
            build_labels_filters(GitHubDonePullRequestFacts, labels, filters, selected, postgres)
        if exclude_inactive:
            date_range = append_activity_days_filter(
                time_from, time_to, selected, filters, ghprt.activity_days, postgres,
            )
        else:
            date_range = set()
        filters.append(
            not_(
                exists().where(
                    ghprt.acc_id == GitHubPullRequestDeployment.acc_id,
                    ghprt.pr_node_id == GitHubPullRequestDeployment.pull_request_id,
                    ghprt.repository_full_name == GitHubPullRequestDeployment.repository_full_name,
                    GitHubPullRequestDeployment.finished_at.between(time_from, time_to),
                ),
            ),
        )
        or_items = or_items()
        selected = sorted(selected, key=lambda i: i.key)
        if postgres:
            return [select(*selected).where(item, *filters) for item in or_items], date_range
        return [select(*selected).where(or_(*or_items), *filters)], date_range

    @classmethod
    async def _compose_query_filters_deployed(
        cls,
        selected: set[InstrumentedAttribute],
        or_items: Callable[[], list[ClauseElement]],
        time_from: Optional[datetime],
        time_to: Optional[datetime],
        participants: PRParticipants,
        labels: LabelFilter,
        exclude_inactive: bool,
        prefixer: Prefixer,
        account: int,
        postgres: bool,
    ) -> list[Select]:
        ghprt = GitHubDonePullRequestFacts
        filters = cls._create_common_filters(None, time_to, None, account)
        selected = selected.copy()
        selected.add(true().label("deployed"))
        if len(participants) > 0:
            await cls._build_participants_filters(
                participants, filters, selected, postgres, prefixer,
            )
        if labels:
            build_labels_filters(GitHubDonePullRequestFacts, labels, filters, selected, postgres)
        filters.append(
            exists().where(
                ghprt.acc_id == GitHubPullRequestDeployment.acc_id,
                ghprt.pr_node_id == GitHubPullRequestDeployment.pull_request_id,
                ghprt.repository_full_name == GitHubPullRequestDeployment.repository_full_name,
                GitHubPullRequestDeployment.finished_at.between(time_from, time_to),
            ),
        )
        if exclude_inactive and not postgres:
            selected.add(ghprt.activity_days)
        or_items = or_items()
        selected = sorted(selected, key=lambda i: i.key)
        if postgres:
            return [select(selected).where(and_(item, *filters)) for item in or_items]
        return [select(selected).where(and_(or_(*or_items), *filters))]

    @classmethod
    def _create_common_filters(
        cls,
        time_from: Optional[datetime],
        time_to: Optional[datetime],
        repos: Optional[Collection[str]],
        account: int,
    ) -> list[ClauseElement]:
        assert isinstance(time_from, (datetime, type(None)))
        assert isinstance(time_to, (datetime, type(None)))
        ghprt = GitHubDonePullRequestFacts
        items = [
            ghprt.acc_id == account,
            ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        ]
        if time_to is not None:
            items.append(ghprt.pr_created_at < time_to)
        if time_from is not None:
            items.append(ghprt.pr_done_at >= time_from)
        if repos is not None:
            items.append(ghprt.repository_full_name.in_(repos))
        return items

    @classmethod
    def _post_process_ambiguous_done_prs(
        cls,
        result: dict[tuple[int, str], Mapping[str, Any]],
        ambiguous: dict[ReleaseMatch, dict[tuple[int, str], Mapping[str, Any]]],
    ) -> tuple[dict[tuple[int, str], Mapping[str, Any]], dict[str, list[int]]]:
        """Figure out what to do with uncertain `tag_or_branch` release matches."""
        result.update(ambiguous[ReleaseMatch.tag.name])
        # we've found PRs released by tag belonging to these repos.
        # this means that we are going to load tags in load_releases().
        confirmed_tag_repos = {repo for _, repo in ambiguous[ReleaseMatch.tag.name]}
        # regarding the rest - we don't know which releases we'll load, so mark such PRs
        # as ambiguous
        ambiguous_prs = defaultdict(list)
        for (node_id, repo), obj in ambiguous[ReleaseMatch.branch.name].items():
            if repo not in confirmed_tag_repos:
                result[(node_id, repo)] = obj
                ambiguous_prs[repo].append(node_id)
        return result, ambiguous_prs

    @classmethod
    async def _build_participants_filters(
        cls,
        participants: PRParticipants,
        filters: list,
        selected: set[InstrumentedAttribute],
        postgres: bool,
        prefixer: Prefixer,
    ) -> None:
        ghdprf = GitHubDonePullRequestFacts
        if postgres:
            dev_conds_single, dev_conds_multiple = await cls._build_participants_conditions(
                participants, prefixer,
            )

            col_parts_dict = defaultdict(list)
            developer_filters_single = []
            for i, (col, col_parts) in enumerate(dev_conds_single):
                developer_filters_single.append(col.in_(col_parts))
                col_parts_dict[col_parts].append(i)
            # do not send the same array several times
            for group in col_parts_dict.values():
                first_val = developer_filters_single[group[0]].right
                for i in group[1:]:
                    developer_filters_single[i].right = first_val

            col_parts_dict.clear()
            developer_filters_multiple = []
            for i, (col, col_parts) in enumerate(dev_conds_multiple):
                developer_filters_multiple.append(col.has_any([str(p) for p in col_parts]))
                col_parts_dict[col_parts].append(i)
            # do not send the same array several times
            for group in col_parts_dict.values():
                first_val = developer_filters_multiple[group[0]].right
                for i in group[1:]:
                    developer_filters_multiple[i].right = first_val

            filters.append(or_(*developer_filters_single, *developer_filters_multiple))
        else:
            selected.update(
                {
                    ghdprf.author,
                    ghdprf.merger,
                    ghdprf.releaser,
                    ghdprf.reviewers,
                    ghdprf.commenters,
                    ghdprf.commit_authors,
                    ghdprf.commit_committers,
                },
            )

    @classmethod
    async def _build_participants_conditions(
        cls,
        participants: PRParticipants,
        prefixer: Prefixer,
    ) -> tuple[list, list]:
        user_login_to_node_get = prefixer.user_login_to_node.get

        def _build_conditions(roles):
            return [
                (c, tuple(chain.from_iterable(user_login_to_node_get(u, []) for u in pset)))
                for c, pset in ((col, participants.get(pk)) for col, pk in roles)
                if pset
            ]

        ghdprf = GitHubDonePullRequestFacts
        single_roles = (
            (ghdprf.author, PRParticipationKind.AUTHOR),
            (ghdprf.merger, PRParticipationKind.MERGER),
            (ghdprf.releaser, PRParticipationKind.RELEASER),
        )
        multiple_roles = (
            (ghdprf.commenters, PRParticipationKind.COMMENTER),
            (ghdprf.reviewers, PRParticipationKind.REVIEWER),
            (ghdprf.commit_authors, PRParticipationKind.COMMIT_AUTHOR),
            (ghdprf.commit_committers, PRParticipationKind.COMMIT_COMMITTER),
        )

        return _build_conditions(single_roles), _build_conditions(multiple_roles)

    @classmethod
    async def _check_participants(
        cls,
        row: Mapping,
        participants: PRParticipants,
        user_node_to_login_get: Callable[[int], str],
    ) -> bool:
        ghprt = GitHubDonePullRequestFacts
        for col, pk in (
            (ghprt.author, PRParticipationKind.AUTHOR),
            (ghprt.merger, PRParticipationKind.MERGER),
            (ghprt.releaser, PRParticipationKind.RELEASER),
        ):
            if user_node_to_login_get(row[col.name]) in participants.get(pk, set()):
                return True
        for col, pk in (
            (ghprt.reviewers, PRParticipationKind.REVIEWER),
            (ghprt.commenters, PRParticipationKind.COMMENTER),
            (ghprt.commit_authors, PRParticipationKind.COMMIT_AUTHOR),
            (ghprt.commit_committers, PRParticipationKind.COMMIT_COMMITTER),
        ):
            devs = {user_node_to_login_get(int(u)) for u in row[col.name]} - {None}
            if devs.intersection(participants.get(pk, set())):
                return True
        return False

    @classmethod
    def _done_pr_facts_from_row(
        cls,
        row: Mapping[str, Any],
        user_node_to_login_get: Callable[[int], str],
        data_name: str,
        pr_node_id_name: str,
        repository_full_name_name: str,
        author_name: str,
        merger_name: str,
        releaser_name: str,
    ) -> PullRequestFacts:
        return PullRequestFacts(
            data=row[data_name],
            node_id=row[pr_node_id_name],
            repository_full_name=row[repository_full_name_name],
            author=user_node_to_login_get(row[author_name], ""),
            merger=user_node_to_login_get(row[merger_name], ""),
            releaser=user_node_to_login_get(row[releaser_name], ""),
        )


@sentry_span
async def store_precomputed_done_facts(
    prs: Iterable[MinedPullRequest],
    pr_facts: Iterable[Optional[PullRequestFacts]],
    time_to: datetime,
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
    account: int,
    pdb: morcilla.Database,
) -> None:
    """Store PullRequestFacts belonging to released or rejected PRs to the precomputed DB."""
    log = logging.getLogger("%s.store_precomputed_done_facts" % metadata.__package__)
    inserted = []
    sqlite = pdb.url.dialect == "sqlite"
    time_to = pd.Timestamp(time_to).to_numpy()
    for pr, facts in zip(prs, pr_facts):
        if facts is None or facts.closed and facts.closed > time_to:
            # ImpossiblePullRequest
            # potentially missing data about a PR
            continue
        pr_created = pr.pr[PullRequest.created_at.name]
        try:
            assert pr_created == facts.created
        except TypeError:
            assert pr_created.to_numpy() == facts.created
        if not facts.released:
            if not (facts.force_push_dropped or (facts.closed and not facts.merged)):
                continue
            done_at = facts.closed.item().replace(tzinfo=timezone.utc)
        else:
            done_at = facts.released.item().replace(tzinfo=timezone.utc)
            if not facts.closed:
                log.error(
                    "[DEV-508] PR %s (%s#%d) is released but not closed:\n%s",
                    pr.pr[PullRequest.node_id.name],
                    pr.pr[PullRequest.repository_full_name.name],
                    pr.pr[PullRequest.number.name],
                    facts,
                )
                continue
        repo = pr.pr[PullRequest.repository_full_name.name]
        if pr.release[matched_by_column] is not None:
            release_match = (
                release_settings.native[repo]
                .with_match(ReleaseMatch(pr.release[matched_by_column]))
                .as_db(default_branches[drop_logical_repo(repo)])
            )
        else:
            release_match = ReleaseMatch.rejected.name
        participants = pr.participant_nodes()
        inserted.append(
            GitHubDonePullRequestFacts(
                acc_id=account,
                pr_node_id=pr.pr[PullRequest.node_id.name],
                release_match=release_match,
                repository_full_name=repo,
                pr_created_at=facts.created.item().replace(tzinfo=timezone.utc),
                pr_done_at=done_at,
                number=pr.pr[PullRequest.number.name],
                release_url=pr.release[Release.url.name],
                release_node_id=pr.release[Release.node_id.name],
                author=_flatten_set(participants[PRParticipationKind.AUTHOR]),
                merger=_flatten_set(participants[PRParticipationKind.MERGER]),
                releaser=_flatten_set(participants[PRParticipationKind.RELEASER]),
                commenters={str(k): "" for k in participants[PRParticipationKind.COMMENTER]},
                reviewers={str(k): "" for k in participants[PRParticipationKind.REVIEWER]},
                commit_authors={
                    str(k): "" for k in participants[PRParticipationKind.COMMIT_AUTHOR]
                },
                commit_committers={
                    str(k): "" for k in participants[PRParticipationKind.COMMIT_COMMITTER]
                },
                labels={label: "" for label in pr.labels[PullRequestLabel.name.name].values},
                activity_days=collect_activity_days(pr, facts, sqlite),
                data=facts.data,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        )
    if not inserted:
        return
    sql = (await dialect_specific_insert(pdb))(GitHubDonePullRequestFacts)
    sql = sql.on_conflict_do_update(
        index_elements=GitHubDonePullRequestFacts.__table__.primary_key.columns,
        set_={
            col.name: getattr(sql.excluded, col.name)
            for col in (
                GitHubDonePullRequestFacts.pr_done_at,
                GitHubDonePullRequestFacts.updated_at,
                GitHubDonePullRequestFacts.release_url,
                GitHubDonePullRequestFacts.release_node_id,
                GitHubDonePullRequestFacts.merger,
                GitHubDonePullRequestFacts.releaser,
                GitHubDonePullRequestFacts.activity_days,
                GitHubDonePullRequestFacts.data,
            )
        },
    )
    with sentry_sdk.start_span(op="store_precomputed_done_facts/execute_many"):
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, inserted)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, inserted)


@sentry_span
async def detect_force_push_dropped_prs(
    repos: Iterable[str],
    branches: pd.DataFrame,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> Collection[str]:
    """
    Load all released precomputed PRs and re-check that they are still accessible from \
    the branch heads. Mark inaccessible as force push dropped.

    We don't try to resolve rebased PRs here due to the intended use case.
    """
    log = logging.getLogger(f"{metadata.__package__}.detect_force_push_dropped_prs")
    ghdprf = GitHubDonePullRequestFacts
    format_version = ghdprf.__table__.columns[ghdprf.format_version.key].default.arg
    prs_df, dags = await gather(
        read_sql_query(
            select(ghdprf.pr_node_id).where(
                ghdprf.acc_id == account,
                ghdprf.format_version == format_version,
                ghdprf.release_match.like("%|%"),
                ghdprf.repository_full_name.in_(repos),
            ),
            pdb,
            [ghdprf.pr_node_id],
        ),
        fetch_precomputed_commit_history_dags(repos, account, pdb, cache),
        op="fetch prs + branches + dags",
    )
    pr_node_ids = prs_df[ghdprf.pr_node_id.name].values
    del prs_df
    node_commit = aliased(NodeCommit, name="c")
    node_pr = aliased(NodePullRequest, name="pr")
    pr_merges, dags = await gather(
        read_sql_query(
            select(node_commit.sha, node_pr.node_id)
            .select_from(
                join(
                    node_pr,
                    node_commit,
                    and_(
                        node_pr.acc_id == node_commit.acc_id,
                        node_pr.merge_commit_id == node_commit.graph_id,
                    ),
                ),
            )
            .where(node_pr.acc_id.in_(meta_ids), node_pr.node_id.in_any_values(pr_node_ids))
            .order_by(node_commit.sha)
            .with_statement_hint("Leading(((*VALUES* pr) c))")
            .with_statement_hint(f"Rows(*VALUES* pr #{len(pr_node_ids)})")
            .with_statement_hint(f"Rows(*VALUES* pr c #{len(pr_node_ids)})"),
            mdb,
            [node_commit.sha, node_pr.node_id],
        ),
        fetch_repository_commits(
            dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, True, account, meta_ids, mdb, pdb, cache,
        ),
        op="fetch merges + prune dags",
    )
    del pr_node_ids
    if dags:
        accessible_hashes = np.unique(np.concatenate([dag[1][0] for dag in dags.values()]))
    else:
        accessible_hashes = np.array([], dtype="S40")
    merge_hashes = pr_merges[node_commit.sha.name].values
    if len(accessible_hashes) > 0:
        found = searchsorted_inrange(accessible_hashes, merge_hashes)
        dead_indexes = np.flatnonzero(accessible_hashes[found] != merge_hashes)
    else:
        log.error("all these repositories have empty commit DAGs: %s", sorted(dags))
        dead_indexes = np.arange(len(merge_hashes))
    dead_pr_node_ids = pr_merges[node_pr.node_id.name].values[dead_indexes]
    if len(dead_indexes) == 0:
        return dead_pr_node_ids
    del pr_merges
    log.info("updating %d force push dropped PRs", len(dead_indexes))
    if await is_postgresql(pdb):
        patch_expr = func.overlay(
            ghdprf.data,
            text("PLACING"),
            b"\x01",
            text("FROM"),
            PullRequestFacts.dtype.fields[PullRequestFacts.f.force_push_dropped][1] + 1,
        )
        patch_expr.clauses.operator = None
        data_patch = {ghdprf.data: patch_expr}
    else:
        # SQLite cannot patch nor concatenate blobs
        data_patch = {}
    ghdprf2 = aliased(GitHubDonePullRequestFacts, name="alive_ghdprf")
    batch_size = 1000
    now = datetime.now(timezone.utc)
    with sentry_sdk.start_span(
        op="update force push dropped prs", description=str(len(dead_indexes)),
    ):
        for batch in range(0, len(dead_pr_node_ids) + batch_size - 1, batch_size):
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute(
                        delete(ghdprf).where(
                            ghdprf.acc_id == account,
                            ghdprf.format_version == format_version,
                            ghdprf.pr_node_id.in_(dead_pr_node_ids[batch : batch + batch_size]),
                            ghdprf.release_match == ReleaseMatch.force_push_drop.name,
                            exists().where(
                                ghdprf2.acc_id == ghdprf.acc_id,
                                ghdprf2.format_version == ghdprf.format_version,
                                ghdprf2.pr_node_id == ghdprf.pr_node_id,
                                ghdprf2.release_match.like("%|%"),
                            ),
                        ),
                    )
                    await pdb_conn.execute(
                        update(ghdprf)
                        .where(
                            ghdprf.acc_id == account,
                            ghdprf.format_version == format_version,
                            ghdprf.pr_node_id.in_(dead_pr_node_ids[batch : batch + batch_size]),
                            ghdprf.release_match != ReleaseMatch.force_push_drop.name,
                        )
                        .values(
                            {
                                ghdprf.updated_at: now,
                                ghdprf.release_match: ReleaseMatch.force_push_drop.name,
                                **data_patch,
                            },
                        ),
                    )

    return dead_pr_node_ids


def _flatten_set(s: set) -> Optional[Any]:
    if not s:
        return None
    assert len(s) == 1
    return next(iter(s))
