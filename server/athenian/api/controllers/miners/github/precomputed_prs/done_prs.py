from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
import logging
import pickle
from typing import Any, Callable, Collection, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import aiomcache
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, delete, exists, insert, not_, or_, select, text, union_all
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement, Select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.controllers.logical_repos import drop_logical_repo
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.branches import load_branch_commit_dates
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    fetch_precomputed_commit_history_dags, fetch_repository_commits
from athenian.api.controllers.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.controllers.miners.github.precomputed_prs.utils import \
    append_activity_days_filter, build_labels_filters, collect_activity_days, \
    labels_are_compatible, triage_by_release_match
from athenian.api.controllers.miners.github.release_load import group_repos_by_release_match, \
    match_groups_to_sql
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.types import MinedPullRequest, PRParticipants, \
    PRParticipationKind, PullRequestFacts, PullRequestFactsMap
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import ReleaseMatch, ReleaseSettings
from athenian.api.db import ParallelDatabase
from athenian.api.models.metadata.github import PullRequest, PullRequestLabel, Release
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubPullRequestDeployment
from athenian.api.tracing import sentry_span


class DonePRFactsLoader:
    """Loader for done PRs facts."""

    @classmethod
    @sentry_span
    async def load_precomputed_done_candidates(cls,
                                               time_from: datetime,
                                               time_to: datetime,
                                               repos: Collection[str],
                                               default_branches: Dict[str, str],
                                               release_settings: ReleaseSettings,
                                               account: int,
                                               pdb: databases.Database,
                                               ) -> Tuple[Set[int], Dict[str, List[int]]]:
        """
        Load the set of done PR identifiers and specifically ambiguous PR node IDs.

        We find all the done PRs for a given time frame, repositories, and release match settings.

        Note: we don't include the deployed PRs!

        :return: 1. Done PR node IDs. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        ghprt = GitHubDonePullRequestFacts
        selected = [ghprt.pr_node_id,
                    ghprt.repository_full_name,
                    ghprt.release_match]
        filters = cls._create_common_filters(time_from, time_to, repos, account)
        with sentry_sdk.start_span(op="load_precomputed_done_candidates/fetch"):
            rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        for row in rows:
            dump = triage_by_release_match(
                row[ghprt.repository_full_name.name], row[ghprt.release_match.name],
                release_settings, default_branches, result, ambiguous)
            if dump is None:
                continue
            dump[(row[ghprt.pr_node_id.name], row[ghprt.repository_full_name.name])] = row
        result, ambiguous = cls._post_process_ambiguous_done_prs(result, ambiguous)
        return {node_id for node_id, _ in result}, ambiguous

    @classmethod
    @sentry_span
    async def load_precomputed_done_facts_filters(cls,
                                                  time_from: datetime,
                                                  time_to: datetime,
                                                  repos: Collection[str],
                                                  participants: PRParticipants,
                                                  labels: LabelFilter,
                                                  default_branches: Dict[str, str],
                                                  exclude_inactive: bool,
                                                  release_settings: ReleaseSettings,
                                                  prefixer: Prefixer,
                                                  account: int,
                                                  pdb: databases.Database,
                                                  ) -> Tuple[PullRequestFactsMap,
                                                             Dict[str, List[int]]]:
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
            time_from, time_to, repos, participants, labels,
            default_branches, exclude_inactive, release_settings, prefixer, account, pdb)
        user_node_to_login_get = prefixer.user_node_to_login.get
        for key, row in result.items():
            result[key] = cls._done_pr_facts_from_row(row, user_node_to_login_get)
        return result, ambiguous

    @classmethod
    async def load_precomputed_done_facts_all(cls,
                                              repos: Collection[str],
                                              default_branches: Dict[str, str],
                                              release_settings: ReleaseSettings,
                                              prefixer: Prefixer,
                                              account: int,
                                              pdb: databases.Database,
                                              extra: Iterable[InstrumentedAttribute] = (),
                                              ) -> Tuple[PullRequestFactsMap,
                                                         Dict[str, Mapping[str, Any]]]:
        """
        Fetch all the precomputed done PR facts we have.

        We don't set the repository, the author, and the merger!

        :param extra: Additional columns to fetch.

        :return: 1. Map from PR node IDs to repo names and facts. \
                 2. Map from PR node IDs to raw returned rows.
        """
        ghdprf = GitHubDonePullRequestFacts
        result, _ = await cls._load_precomputed_done_filters(
            [ghdprf.data, ghdprf.releaser, *extra],
            None, None, repos, {}, LabelFilter.empty(),
            default_branches, False, release_settings, prefixer, account, pdb)
        raw = {}
        user_node_to_login_get = prefixer.user_node_to_login.get
        for (node_id, repo), row in result.items():
            result[(node_id, repo)] = PullRequestFacts(
                data=row[ghdprf.data.name],
                releaser=user_node_to_login_get(row[ghdprf.releaser.name]))
            raw[node_id] = row
        return result, raw

    @classmethod
    @sentry_span
    async def load_precomputed_done_timestamp_filters(cls,
                                                      time_from: datetime,
                                                      time_to: datetime,
                                                      repos: Collection[str],
                                                      participants: PRParticipants,
                                                      labels: LabelFilter,
                                                      default_branches: Dict[str, str],
                                                      exclude_inactive: bool,
                                                      release_settings: ReleaseSettings,
                                                      prefixer: Prefixer,
                                                      account: int,
                                                      pdb: databases.Database,
                                                      ) -> Tuple[Dict[Tuple[int, str], datetime],
                                                                 Dict[str, List[int]]]:
        """
        Fetch precomputed done PR "pr_done_at" timestamps.

        :return: 1. map from PR node IDs to their release timestamps. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        prs, ambiguous = await cls._load_precomputed_done_filters(
            [GitHubDonePullRequestFacts.pr_done_at], time_from, time_to, repos, participants,
            labels, default_branches, exclude_inactive, release_settings, prefixer, account, pdb)
        result = {
            key: row[GitHubDonePullRequestFacts.pr_done_at.name]
            for key, row in prs.items()
        }
        if pdb.url.dialect == "sqlite":
            for key, dt in result.items():
                result[key] = dt.replace(tzinfo=timezone.utc)
        return result, ambiguous

    @classmethod
    @sentry_span
    async def load_precomputed_done_facts_reponums(cls,
                                                   repos: Dict[str, Set[int]],
                                                   default_branches: Dict[str, str],
                                                   release_settings: ReleaseSettings,
                                                   prefixer: Prefixer,
                                                   account: int,
                                                   pdb: databases.Database,
                                                   ) -> Tuple[PullRequestFactsMap,
                                                              Dict[str, List[int]]]:
        """
        Load PullRequestFacts belonging to released or rejected PRs from the precomputed DB.

        repo + numbers version.

        :return: 1. Map PR node ID -> repository name & specified column value. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        ghprt = GitHubDonePullRequestFacts
        selected = [ghprt.pr_node_id,
                    ghprt.repository_full_name,
                    ghprt.release_match,
                    ghprt.data,
                    ghprt.author,
                    ghprt.merger,
                    ghprt.releaser,
                    ]
        format_version_filter = \
            ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg
        if pdb.url.dialect == "sqlite":
            filters = [
                format_version_filter,
                or_(*[and_(ghprt.repository_full_name == repo,
                           ghprt.number.in_(numbers),
                           ghprt.acc_id == account)
                      for repo, numbers in repos.items()]),
            ]
            query = select(selected).where(and_(*filters))
        else:
            match_groups, event_repos, _ = group_repos_by_release_match(
                repos, default_branches, release_settings)
            match_groups[ReleaseMatch.rejected] = match_groups[ReleaseMatch.force_push_drop] = \
                {"": repos}
            if event_repos:
                match_groups[ReleaseMatch.event] = {"": event_repos}
            or_items, or_repos = match_groups_to_sql(match_groups, ghprt)
            query = union_all(*(
                select(selected).where(and_(item, format_version_filter, or_(
                    *[and_(ghprt.repository_full_name == repo,
                           ghprt.number.in_(repos[repo]),
                           ghprt.acc_id == account)
                      for repo in item_repos],
                )))
                for item, item_repos in zip(or_items, or_repos)))

        with sentry_sdk.start_span(op="load_precomputed_done_facts_reponums/fetch"):
            rows = await pdb.fetch_all(query)
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        user_node_to_login_get = prefixer.user_node_to_login.get
        for row in rows:
            repo = row[ghprt.repository_full_name.name]
            dump = triage_by_release_match(
                repo, row[ghprt.release_match.name],
                release_settings, default_branches, result, ambiguous)
            if dump is None:
                continue
            dump[(row[ghprt.pr_node_id.name], repo)] = cls._done_pr_facts_from_row(
                row, user_node_to_login_get)
        return cls._post_process_ambiguous_done_prs(result, ambiguous)

    @classmethod
    @sentry_span
    async def load_precomputed_done_facts_ids(cls,
                                              node_ids: Iterable[int],
                                              default_branches: Dict[str, str],
                                              release_settings: ReleaseSettings,
                                              prefixer: Prefixer,
                                              account: int,
                                              pdb: databases.Database,
                                              panic_on_missing_repositories: bool = True,
                                              ) -> Tuple[PullRequestFactsMap,
                                                         Dict[str, List[int]]]:
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
        selected = [ghprt.pr_node_id,
                    ghprt.repository_full_name,
                    ghprt.release_match,
                    ghprt.data,
                    ghprt.author,
                    ghprt.merger,
                    ghprt.releaser,
                    ]
        filters = [
            ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
            ghprt.pr_node_id.in_(node_ids),
            ghprt.acc_id == account,
        ]
        query = select(selected).where(and_(*filters))
        with sentry_sdk.start_span(op="load_precomputed_done_facts_ids/fetch"):
            rows = await pdb.fetch_all(query)
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        user_node_to_login_get = prefixer.user_node_to_login.get
        for row in rows:
            repo = row[ghprt.repository_full_name.name]
            if not panic_on_missing_repositories and repo not in release_settings.native:
                log.warning("Discarding PR %s because repository %s is missing",
                            row[ghprt.pr_node_id.name], repo)
                continue
            dump = triage_by_release_match(
                repo, row[ghprt.release_match.name],
                release_settings, default_branches, result, ambiguous)
            if dump is None:
                continue
            dump[(row[ghprt.pr_node_id.name], repo)] = \
                cls._done_pr_facts_from_row(row, user_node_to_login_get)
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
    async def load_precomputed_pr_releases(cls,
                                           prs: pd.DataFrame,
                                           time_to: datetime,
                                           matched_bys: Dict[str, ReleaseMatch],
                                           default_branches: Dict[str, str],
                                           release_settings: ReleaseSettings,
                                           prefixer: Prefixer,
                                           account: int,
                                           pdb: databases.Database,
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
        with sentry_sdk.start_span(op="load_precomputed_pr_releases/fetch"):
            prs = await pdb.fetch_all(
                select([ghprt.pr_node_id, ghprt.pr_done_at, ghprt.releaser, ghprt.release_url,
                        ghprt.release_node_id, ghprt.repository_full_name, ghprt.release_match])
                .where(and_(ghprt.pr_node_id.in_(prs.index.get_level_values(0).values),
                            ghprt.repository_full_name.in_(
                                prs.index.get_level_values(1).unique()),
                            ghprt.acc_id == account,
                            ghprt.releaser.isnot(None),
                            ghprt.pr_done_at < time_to)))
        records = []
        utc = timezone.utc
        force_push_dropped = set()
        user_node_to_login_get = prefixer.user_node_to_login.get
        for pr in prs:
            repo = pr[ghprt.repository_full_name.name]
            node_id = pr[ghprt.pr_node_id.name]
            release_match = pr[ghprt.release_match.name]
            author_node_id = pr[ghprt.releaser.name]
            if release_match in (ReleaseMatch.force_push_drop.name, ReleaseMatch.event.name):
                if release_match == ReleaseMatch.force_push_drop.name:
                    if node_id in force_push_dropped:
                        continue
                    force_push_dropped.add(node_id)
                records.append((node_id,
                                pr[ghprt.pr_done_at.name].replace(tzinfo=utc),
                                user_node_to_login_get(author_node_id),
                                author_node_id,
                                pr[ghprt.release_url.name],
                                pr[ghprt.release_node_id.name],
                                pr[ghprt.repository_full_name.name],
                                ReleaseMatch[release_match]))
                continue
            try:
                release_setting = release_settings.native[repo].with_match(matched_bys[repo])
            except KeyError:
                # pdb thinks this PR was released but our current release matching settings
                # disagree
                log.warning("Alternative release matching detected: %s", dict(pr))
                continue
            if not release_setting.compatible_with_db(
                    release_match, default_branches[drop_logical_repo(repo)]):
                continue
            records.append((node_id,
                            pr[ghprt.pr_done_at.name].replace(tzinfo=utc),
                            user_node_to_login_get(author_node_id),
                            author_node_id,
                            pr[ghprt.release_url.name],
                            pr[ghprt.release_node_id.name],
                            pr[ghprt.repository_full_name.name],
                            release_setting.match))
        return new_released_prs_df(records)

    @classmethod
    @sentry_span
    async def _load_precomputed_done_filters(cls,
                                             columns: List[InstrumentedAttribute],
                                             time_from: Optional[datetime],
                                             time_to: Optional[datetime],
                                             repos: Collection[str],
                                             participants: PRParticipants,
                                             labels: LabelFilter,
                                             default_branches: Dict[str, str],
                                             exclude_inactive: bool,
                                             release_settings: ReleaseSettings,
                                             prefixer: Prefixer,
                                             account: int,
                                             pdb: databases.Database,
                                             ) -> Tuple[Dict[Tuple[int, str], Mapping[str, Any]],
                                                        Dict[str, List[int]]]:
        """
        Load some data belonging to released or rejected PRs from the precomputed DB.

        Query version. JIRA must be filtered separately.
        :return: 1. Map (PR node ID, repository name) -> facts. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        postgres = pdb.url.dialect == "postgresql"
        ghprt = GitHubDonePullRequestFacts
        selected = [
            ghprt.pr_node_id,
            ghprt.repository_full_name,
            ghprt.release_match,
        ] + columns
        match_groups, event_repos, _ = group_repos_by_release_match(
            repos, default_branches, release_settings)
        match_groups[ReleaseMatch.rejected] = match_groups[ReleaseMatch.force_push_drop] = \
            {"": list(repos)}
        if event_repos:
            match_groups[ReleaseMatch.event] = {"": event_repos}

        def or_items():
            return match_groups_to_sql(match_groups, ghprt)[0]

        queries_undeployed, date_range = await cls._compose_query_filters_undeployed(
            selected, or_items, time_from, time_to, participants, labels, exclude_inactive,
            prefixer, account, postgres)
        queries_deployed = await cls._compose_query_filters_deployed(
            selected, or_items, time_from, time_to, participants, labels, exclude_inactive,
            prefixer, account, postgres)
        query = union_all(*(queries_undeployed + queries_deployed))
        with sentry_sdk.start_span(op="_load_precomputed_done_filters/fetch"):
            rows = await pdb.fetch_all(query)
        result = {}
        ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
        if labels and not postgres:
            include_singles, include_multiples = LabelFilter.split(labels.include)
            include_singles = set(include_singles)
            include_multiples = [set(m) for m in include_multiples]
        if len(participants) > 0 and not postgres:
            user_node_to_login_get = prefixer.user_node_to_login.get
        for row in rows:
            repo, rm = row[ghprt.repository_full_name.name], row[ghprt.release_match.name]
            dump = triage_by_release_match(
                repo, rm, release_settings, default_branches, result, ambiguous)
            if dump is None:
                continue
            if not postgres:
                if len(participants) > 0 and not await cls._check_participants(
                        row, participants, user_node_to_login_get):
                    continue
                if labels and not labels_are_compatible(include_singles, include_multiples,
                                                        labels.exclude, row[ghprt.labels.name]):
                    continue
                if exclude_inactive and not row["deployed"]:
                    activity_days = {datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                     for d in row[ghprt.activity_days.name]}
                    if not activity_days.intersection(date_range):
                        continue
            dump[(row[ghprt.pr_node_id.name], repo)] = row
        return cls._post_process_ambiguous_done_prs(result, ambiguous)

    @classmethod
    async def _compose_query_filters_undeployed(cls,
                                                selected: List[InstrumentedAttribute],
                                                or_items: Callable[[], List[ClauseElement]],
                                                time_from: Optional[datetime],
                                                time_to: Optional[datetime],
                                                participants: PRParticipants,
                                                labels: LabelFilter,
                                                exclude_inactive: bool,
                                                prefixer: Prefixer,
                                                account: int,
                                                postgres: bool,
                                                ) -> Tuple[List[Select], Set[datetime]]:
        ghprt = GitHubDonePullRequestFacts
        filters = cls._create_common_filters(time_from, time_to, None, account)
        selected = selected.copy()
        selected.append(text("false AS deployed"))
        if len(participants) > 0:
            await cls._build_participants_filters(
                participants, filters, selected, postgres, prefixer)
        if labels:
            build_labels_filters(GitHubDonePullRequestFacts, labels, filters, selected, postgres)
        if exclude_inactive:
            date_range = append_activity_days_filter(
                time_from, time_to, selected, filters, ghprt.activity_days, postgres)
        else:
            date_range = set()
        filters.append(not_(exists().where(and_(
            ghprt.acc_id == GitHubPullRequestDeployment.acc_id,
            ghprt.pr_node_id == GitHubPullRequestDeployment.pull_request_id,
            GitHubPullRequestDeployment.finished_at.between(time_from, time_to),
        ))))
        or_items = or_items()
        if postgres:
            return [select(selected).where(and_(item, *filters)) for item in or_items], date_range
        return [select(selected).where(and_(or_(*or_items), *filters))], date_range

    @classmethod
    async def _compose_query_filters_deployed(cls,
                                              selected: List[InstrumentedAttribute],
                                              or_items: Callable[[], List[ClauseElement]],
                                              time_from: Optional[datetime],
                                              time_to: Optional[datetime],
                                              participants: PRParticipants,
                                              labels: LabelFilter,
                                              exclude_inactive: bool,
                                              prefixer: Prefixer,
                                              account: int,
                                              postgres: bool,
                                              ) -> List[Select]:
        ghprt = GitHubDonePullRequestFacts
        filters = cls._create_common_filters(None, time_to, None, account)
        selected = selected.copy()
        selected.append(text("true AS deployed"))
        if len(participants) > 0:
            await cls._build_participants_filters(
                participants, filters, selected, postgres, prefixer)
        if labels:
            build_labels_filters(GitHubDonePullRequestFacts, labels, filters, selected, postgres)
        filters.append(exists().where(and_(
            ghprt.acc_id == GitHubPullRequestDeployment.acc_id,
            ghprt.pr_node_id == GitHubPullRequestDeployment.pull_request_id,
            GitHubPullRequestDeployment.finished_at.between(time_from, time_to),
        )))
        if exclude_inactive and not postgres:
            selected.append(ghprt.activity_days)
        or_items = or_items()
        if postgres:
            return [select(selected).where(and_(item, *filters)) for item in or_items]
        return [select(selected).where(and_(or_(*or_items), *filters))]

    @classmethod
    def _create_common_filters(cls,
                               time_from: Optional[datetime],
                               time_to: Optional[datetime],
                               repos: Optional[Collection[str]],
                               account: int,
                               ) -> List[ClauseElement]:
        assert isinstance(time_from, (datetime, type(None)))
        assert isinstance(time_to, (datetime, type(None)))
        ghprt = GitHubDonePullRequestFacts
        items = [
            ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
            ghprt.acc_id == account,
        ]
        if time_to is not None:
            items.append(ghprt.pr_created_at < time_to)
        if time_from is not None:
            items.append(ghprt.pr_done_at >= time_from)
        if repos is not None:
            items.append(ghprt.repository_full_name.in_(repos))
        return items

    @classmethod
    def _post_process_ambiguous_done_prs(cls,
                                         result: Dict[Tuple[int, str], Mapping[str, Any]],
                                         ambiguous: Dict[ReleaseMatch,
                                                         Dict[Tuple[int, str], Mapping[str, Any]]],
                                         ) -> Tuple[Dict[Tuple[int, str], Mapping[str, Any]],
                                                    Dict[str, List[int]]]:
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
    async def _build_participants_filters(cls,
                                          participants: PRParticipants,
                                          filters: list,
                                          selected: list,
                                          postgres: bool,
                                          prefixer: Prefixer,
                                          ) -> None:
        ghdprf = GitHubDonePullRequestFacts
        if postgres:
            dev_conds_single, dev_conds_multiple = \
                await cls._build_participants_conditions(participants, prefixer)

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
            selected.extend([
                ghdprf.author, ghdprf.merger, ghdprf.releaser, ghdprf.reviewers, ghdprf.commenters,
                ghdprf.commit_authors, ghdprf.commit_committers])

    @classmethod
    async def _build_participants_conditions(cls,
                                             participants: PRParticipants,
                                             prefixer: Prefixer,
                                             ) -> Tuple[list, list]:
        user_login_to_node_get = prefixer.user_login_to_node.get

        def _build_conditions(roles):
            return [
                (c, tuple(chain.from_iterable(user_login_to_node_get(u, []) for u in pset)))
                for c, pset in ((col, participants.get(pk)) for col, pk in roles)
                if pset
            ]

        ghdprf = GitHubDonePullRequestFacts
        single_roles = ((ghdprf.author, PRParticipationKind.AUTHOR),
                        (ghdprf.merger, PRParticipationKind.MERGER),
                        (ghdprf.releaser, PRParticipationKind.RELEASER))
        multiple_roles = ((ghdprf.commenters, PRParticipationKind.COMMENTER),
                          (ghdprf.reviewers, PRParticipationKind.REVIEWER),
                          (ghdprf.commit_authors, PRParticipationKind.COMMIT_AUTHOR),
                          (ghdprf.commit_committers, PRParticipationKind.COMMIT_COMMITTER))

        return _build_conditions(single_roles), _build_conditions(multiple_roles)

    @classmethod
    async def _check_participants(cls,
                                  row: Mapping,
                                  participants: PRParticipants,
                                  user_node_to_login_get: Callable[[int], str],
                                  ) -> bool:
        ghprt = GitHubDonePullRequestFacts
        for col, pk in ((ghprt.author, PRParticipationKind.AUTHOR),
                        (ghprt.merger, PRParticipationKind.MERGER),
                        (ghprt.releaser, PRParticipationKind.RELEASER)):
            if user_node_to_login_get(row[col.name]) in participants.get(pk, set()):
                return True
        for col, pk in ((ghprt.reviewers, PRParticipationKind.REVIEWER),
                        (ghprt.commenters, PRParticipationKind.COMMENTER),
                        (ghprt.commit_authors, PRParticipationKind.COMMIT_AUTHOR),
                        (ghprt.commit_committers, PRParticipationKind.COMMIT_COMMITTER)):
            devs = {user_node_to_login_get(int(u)) for u in row[col.name]} - {None}
            if devs.intersection(participants.get(pk, set())):
                return True
        return False

    @classmethod
    def _done_pr_facts_from_row(cls,
                                row: Mapping[str, Any],
                                user_node_to_login_get: Callable[[int], str],
                                ) -> PullRequestFacts:
        ghdprf = GitHubDonePullRequestFacts
        return PullRequestFacts(
            data=row[ghdprf.data.name],
            repository_full_name=row[ghdprf.repository_full_name.name],
            author=user_node_to_login_get(row[ghdprf.author.name]),
            merger=user_node_to_login_get(row[ghdprf.merger.name]),
            releaser=user_node_to_login_get(row[ghdprf.releaser.name]))


@sentry_span
async def store_precomputed_done_facts(prs: Iterable[MinedPullRequest],
                                       pr_facts: Iterable[Optional[PullRequestFacts]],
                                       default_branches: Dict[str, str],
                                       release_settings: ReleaseSettings,
                                       account: int,
                                       pdb: databases.Database,
                                       ) -> None:
    """Store PullRequestFacts belonging to released or rejected PRs to the precomputed DB."""
    log = logging.getLogger("%s.store_precomputed_done_facts" % metadata.__package__)
    inserted = []
    sqlite = pdb.url.dialect == "sqlite"
    for pr, facts in zip(prs, pr_facts):
        if facts is None:
            # ImpossiblePullRequest
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
                log.error("[DEV-508] PR %s (%s#%d) is released but not closed:\n%s",
                          pr.pr[PullRequest.node_id.name],
                          pr.pr[PullRequest.repository_full_name.name],
                          pr.pr[PullRequest.number.name],
                          facts)
                continue
        repo = pr.pr[PullRequest.repository_full_name.name]
        if pr.release[matched_by_column] is not None:
            release_match = release_settings.native[repo] \
                .with_match(ReleaseMatch(pr.release[matched_by_column])) \
                .as_db(default_branches[drop_logical_repo(repo)])
        else:
            release_match = ReleaseMatch.rejected.name
        participants = pr.participant_nodes()
        inserted.append(GitHubDonePullRequestFacts(
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
            commit_authors={str(k): "" for k in participants[PRParticipationKind.COMMIT_AUTHOR]},
            commit_committers={
                str(k): "" for k in participants[PRParticipationKind.COMMIT_COMMITTER]
            },
            labels={label: "" for label in pr.labels[PullRequestLabel.name.name].values},
            activity_days=collect_activity_days(pr, facts, sqlite),
            data=facts.data,
        ).create_defaults().explode(with_primary_keys=True))
    if not inserted:
        return
    if pdb.url.dialect == "postgresql":
        sql = postgres_insert(GitHubDonePullRequestFacts)
        sql = sql.on_conflict_do_update(
            constraint=GitHubDonePullRequestFacts.__table__.primary_key,
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
    elif pdb.url.dialect == "sqlite":
        sql = insert(GitHubDonePullRequestFacts).prefix_with("OR REPLACE")
    else:
        raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
    with sentry_sdk.start_span(op="store_precomputed_done_facts/execute_many"):
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, inserted)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, inserted)


@sentry_span
async def delete_force_push_dropped_prs(repos: Iterable[str],
                                        branches: pd.DataFrame,
                                        account: int,
                                        meta_ids: Tuple[int, ...],
                                        mdb: ParallelDatabase,
                                        pdb: ParallelDatabase,
                                        cache: Optional[aiomcache.Client],
                                        ) -> Collection[str]:
    """
    Load all released precomputed PRs and re-check that they are still accessible from \
    the branch heads. Mark inaccessible as force push dropped.

    We don't try to resolve rebased PRs here due to the intended use case.
    """
    ghdprf = GitHubDonePullRequestFacts
    tasks = [
        pdb.fetch_all(select([ghdprf.pr_node_id])
                      .where(and_(ghdprf.repository_full_name.in_(repos),
                                  ghdprf.acc_id == account,
                                  ghdprf.release_match.like("%|%")))),
        load_branch_commit_dates(branches, meta_ids, mdb),
        fetch_precomputed_commit_history_dags(repos, account, pdb, cache),
    ]
    rows, _, dags = await gather(*tasks, op="fetch prs + branches + dags")
    pr_node_ids = [r[0] for r in rows]
    del rows
    tasks = [
        mdb.fetch_all(select([PullRequest.merge_commit_sha, PullRequest.node_id])
                      .where(and_(PullRequest.node_id.in_(pr_node_ids),
                                  PullRequest.acc_id.in_(meta_ids)))
                      .order_by(PullRequest.merge_commit_sha)),
        fetch_repository_commits(
            dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, True, account, meta_ids,
            mdb, pdb, cache),
    ]
    del pr_node_ids
    pr_merges, dags = await gather(*tasks, op="fetch merges + prune dags")
    accessible_hashes = np.sort(np.concatenate([dag[0] for dag in dags.values()]))
    merge_hashes = np.fromiter((r[0] for r in pr_merges), "S40", len(pr_merges))
    if len(accessible_hashes) > 0:
        found = searchsorted_inrange(accessible_hashes, merge_hashes)
        dead_indexes = np.nonzero(accessible_hashes[found] != merge_hashes)[0]
    else:
        log = logging.getLogger(f"{metadata.__package__}.delete_force_push_dropped_prs")
        log.error("all these repositories have empty commit DAGs: %s", sorted(dags))
        dead_indexes = np.arange(len(merge_hashes))
    dead_pr_node_ids = [None] * len(dead_indexes)
    for i, dead_index in enumerate(dead_indexes):
        dead_pr_node_ids[i] = pr_merges[dead_index][1]
    del pr_merges
    with sentry_sdk.start_span(op="delete force push dropped prs",
                               description=str(len(dead_indexes))):
        await pdb.execute(
            delete(ghdprf)
            .where(and_(ghdprf.pr_node_id.in_(dead_pr_node_ids),
                        ghdprf.release_match != ReleaseMatch.force_push_drop.name)))
    return dead_pr_node_ids


def _flatten_set(s: set) -> Optional[Any]:
    if not s:
        return None
    assert len(s) == 1
    return next(iter(s))
