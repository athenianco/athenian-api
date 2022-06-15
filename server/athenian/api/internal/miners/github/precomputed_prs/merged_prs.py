import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
from typing import Collection, Dict, Iterable, KeysView, List, Optional, Set, Tuple, Union

import aiomcache
import morcilla
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import (
    and_,
    desc,
    exists,
    false,
    insert,
    join,
    not_,
    select,
    true,
    union_all,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.sql.functions import coalesce

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.db import add_pdb_hits, greatest
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.filters import LabelFilter
from athenian.api.internal.miners.github.precomputed_prs.utils import (
    append_activity_days_filter,
    build_labels_filters,
    collect_activity_days,
    extract_release_match,
    labels_are_compatible,
    triage_by_release_match,
)
from athenian.api.internal.miners.types import (
    MinedPullRequest,
    PRParticipants,
    PRParticipationKind,
    PullRequestFacts,
    PullRequestFactsMap,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import PullRequest, Release
from athenian.api.models.precomputed.models import (
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubPullRequestDeployment,
)
from athenian.api.tracing import sentry_span


class MergedPRFactsLoader:
    """Loader for merged PRs facts."""

    @classmethod
    @sentry_span
    async def load_merged_unreleased_pull_request_facts(
        cls,
        prs: Union[pd.DataFrame, pd.Index],
        time_to: datetime,
        labels: LabelFilter,
        matched_bys: Dict[str, ReleaseMatch],
        default_branches: Dict[str, str],
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
        time_from: Optional[datetime] = None,
        exclude_inactive: bool = False,
    ) -> PullRequestFactsMap:
        """
        Load the mapping from PR node identifiers which we are sure are not released in one of \
        `releases` to the serialized facts.

        For each merged PR we maintain the set of releases that do include that PR.

        :return: Map from PR node IDs to their facts.
        """
        if time_to != time_to:
            return {}
        assert time_to.tzinfo is not None
        if exclude_inactive:
            assert time_from is not None
        log = logging.getLogger(
            "%s.load_merged_unreleased_pull_request_facts" % metadata.__package__,
        )
        ghmprf = GitHubMergedPullRequestFacts
        postgres = pdb.url.dialect == "postgresql"
        selected = {
            ghmprf.pr_node_id,
            ghmprf.repository_full_name,
            ghmprf.data,
            ghmprf.author,
            ghmprf.merger,
        }
        default_version = ghmprf.__table__.columns[ghmprf.format_version.name].default.arg
        repos_by_match = defaultdict(list)
        if isinstance(prs, pd.Index):
            prs_index = prs
            assert prs.nlevels == 2
        else:
            prs_index = prs.index
        if prs_index.nlevels == 2:
            nodes_col = prs_index.get_level_values(0).values
            repos_col = prs_index.get_level_values(1).values
        else:
            nodes_col = prs_index.values
            repos_col = prs[PullRequest.repository_full_name.name].values
        logical_repos, index_map, counts = np.unique(
            repos_col, return_inverse=True, return_counts=True,
        )
        pr_ids_by_repo = dict(
            zip(logical_repos, np.split(nodes_col[np.argsort(index_map)], np.cumsum(counts[:-1]))),
        )
        for repo in pr_ids_by_repo:
            if (
                release_match := extract_release_match(
                    repo, matched_bys, default_branches, release_settings,
                )
            ) is None:
                # no new releases
                continue
            repos_by_match[release_match].append(repo)

        def compose_common_filters(deployed: bool):
            nonlocal selected
            my_selected = selected.copy()
            my_selected.add([false(), true()][deployed].label("deployed"))
            filters = [
                ghmprf.checked_until >= time_to,
                ghmprf.format_version == default_version,
                ghmprf.acc_id == account,
            ]
            if labels:
                build_labels_filters(ghmprf, labels, filters, my_selected, postgres)
            if exclude_inactive:
                if not deployed:
                    date_range = append_activity_days_filter(
                        time_from, time_to, my_selected, filters, ghmprf.activity_days, postgres,
                    )
                else:
                    date_range = set()
                    if not postgres:
                        my_selected.add(ghmprf.activity_days)
            else:
                date_range = set()
            ghprd = GitHubPullRequestDeployment
            if not deployed:
                filters.append(
                    not_(
                        exists().where(
                            and_(
                                ghmprf.acc_id == ghprd.acc_id,
                                ghmprf.pr_node_id == ghprd.pull_request_id,
                                ghmprf.repository_full_name == ghprd.repository_full_name,
                                ghprd.finished_at < time_to,
                            ),
                        ),
                    ),
                )
            else:
                filters.append(
                    exists().where(
                        and_(
                            ghmprf.acc_id == ghprd.acc_id,
                            ghmprf.pr_node_id == ghprd.pull_request_id,
                            ghmprf.repository_full_name == ghprd.repository_full_name,
                            ghprd.finished_at.between(time_from, time_to)
                            if exclude_inactive
                            else ghprd.finished_at < time_to,
                        ),
                    ),
                )
            return my_selected, filters, date_range

        queries = []
        for deployed in (False, True):
            my_selected, common_filters, deployed_date_range = compose_common_filters(deployed)
            if not deployed:
                date_range = deployed_date_range
            for release_match, repos in repos_by_match.items():
                if not postgres:
                    # SQLite does not support parameter re-usage
                    my_selected, common_filters, deployed_date_range = compose_common_filters(
                        deployed,
                    )
                pr_ids = np.unique(np.concatenate([pr_ids_by_repo[r] for r in repos]))
                filters = [
                    ghmprf.pr_node_id.in_(pr_ids),
                    ghmprf.repository_full_name.in_(repos),
                    ghmprf.release_match == release_match,
                    *common_filters,
                ]
                my_selected = sorted(my_selected, key=lambda i: i.key)
                queries.append(select(my_selected).where(and_(*filters)))
        if not queries:
            return {}
        query = union_all(*queries)
        with sentry_sdk.start_span(op="load_merged_unreleased_pr_facts/fetch"):
            rows = await pdb.fetch_all(query)
        if labels:
            include_singles, include_multiples = LabelFilter.split(labels.include)
            include_singles = set(include_singles)
            include_multiples = [set(m) for m in include_multiples]
        for repo, pr_ids in pr_ids_by_repo.items():
            pr_ids_by_repo[repo] = set(pr_ids)
        facts = {}
        user_node_map_get = prefixer.user_node_to_login.get
        missing_facts = []
        for row in rows:
            if exclude_inactive and not postgres and not row["deployed"]:
                activity_days = {
                    datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    for d in row[ghmprf.activity_days.name]
                }
                if not activity_days.intersection(date_range):
                    continue
            node_id = row[ghmprf.pr_node_id.name]
            data = row[ghmprf.data.name]
            if data is None:
                # There are two known cases:
                # 1. When we load all PRs without a blacklist (/filter/pull_requests) so some
                #    merged PR is matched to releases but exists in
                #    `github_done_pull_request_facts`.
                # 2. "Impossible" PRs that are merged.
                missing_facts.append(node_id)
                continue
            if labels and not labels_are_compatible(
                include_singles, include_multiples, labels.exclude, row[ghmprf.labels.name],
            ):
                continue
            if node_id in pr_ids_by_repo[(repo := row[ghmprf.repository_full_name.name])]:
                facts[(node_id, repo)] = PullRequestFacts(
                    data=data,
                    node_id=node_id,
                    repository_full_name=repo,
                    author=user_node_map_get(row[ghmprf.author.name], ""),
                    merger=user_node_map_get(row[ghmprf.merger.name], ""),
                    releaser="",
                )
        if missing_facts:
            log.warning("No precomputed facts for merged %s", missing_facts)
        return facts

    @classmethod
    @sentry_span
    async def load_merged_pull_request_facts_all(
        cls,
        repos: Collection[str],
        pr_node_id_blacklist: Collection[int],
        account: int,
        pdb: morcilla.Database,
    ) -> PullRequestFactsMap:
        """
        Load the precomputed merged PR facts through all the time.

        We do not load the repository, the author, and the merger!

        :return: Map from PR node IDs to their facts.
        """
        log = logging.getLogger("%s.load_merged_pull_request_facts_all" % metadata.__package__)
        ghmprf = GitHubMergedPullRequestFacts
        selected = [
            ghmprf.pr_node_id,
            ghmprf.repository_full_name,
            ghmprf.data,
        ]
        default_version = ghmprf.__table__.columns[ghmprf.format_version.name].default.arg
        filters = [
            ghmprf.pr_node_id.notin_(pr_node_id_blacklist),
            ghmprf.repository_full_name.in_(repos),
            ghmprf.format_version == default_version,
            ghmprf.acc_id == account,
        ]
        query = select(selected).where(and_(*filters))
        with sentry_sdk.start_span(op="load_merged_pull_request_facts_all/fetch"):
            rows = await pdb.fetch_all(query)
        facts = {}
        missing_facts = []
        for row in rows:
            if (node_id := row[ghmprf.pr_node_id.name]) in facts:
                # different release match settings, we don't care because the facts are the same
                continue
            data = row[ghmprf.data.name]
            if data is None:
                # There are two known cases:
                # 1. When we load all PRs without a blacklist (/filter/pull_requests) so some
                #    merged PR is matched to releases but exists in
                #    `github_done_pull_request_facts`.
                # 2. "Impossible" PRs that are merged.
                missing_facts.append(node_id)
                continue
            repo = row[ghmprf.repository_full_name.name]
            facts[(node_id, repo)] = PullRequestFacts(
                data, node_id=node_id, repository_full_name=repo,
            )
        if missing_facts:
            log.warning("No precomputed facts on account %d for merged %s", account, missing_facts)
        return facts


@sentry_span
async def update_unreleased_prs(
    merged_prs: pd.DataFrame,
    released_prs: pd.DataFrame,
    time_to: datetime,
    labels: Dict[int, List[str]],
    matched_bys: Dict[str, ReleaseMatch],
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    account: int,
    pdb: morcilla.Database,
    unreleased_prs_event: asyncio.Event,
) -> None:
    """
    Bump the last check timestamps for unreleased merged PRs.

    :param merged_prs: Merged PRs to update in the pdb.
    :param released_prs: Released PRs among `merged_prs`. They should be marked as checked until \
                         the release publish time.
    :param time_to: Until when we checked releases for the specified PRs.
    """
    assert time_to.tzinfo is not None
    assert merged_prs.index.nlevels == 2
    time_to = min(time_to, datetime.now(timezone.utc))
    postgres = pdb.url.dialect == "postgresql"
    values = []
    if not released_prs.empty:
        assert released_prs.index.nlevels == 2
        release_times = dict(
            zip(
                released_prs.index.values,
                released_prs[Release.published_at.name] - timedelta(minutes=1),
            ),
        )
    else:
        release_times = {}
    with sentry_sdk.start_span(op="update_unreleased_prs/generate"):
        for repo, repo_prs in merged_prs.groupby(level=1, sort=False):
            if (
                release_match := extract_release_match(
                    repo, matched_bys, default_branches, release_settings,
                )
            ) is None:
                # no new releases
                continue
            for node_id, merged_at, author, merger in zip(
                repo_prs.index.get_level_values(0).values,
                repo_prs[PullRequest.merged_at.name],
                repo_prs[PullRequest.user_node_id.name].values,
                repo_prs[PullRequest.merged_by_id.name].values,
            ):
                try:
                    released_time = release_times[(node_id, repo)]
                except KeyError:
                    checked_until = time_to
                else:
                    if released_time == released_time:
                        checked_until = min(time_to, released_time - timedelta(seconds=1))
                    else:
                        checked_until = merged_at  # force_push_drop
                values.append(
                    GitHubMergedPullRequestFacts(
                        acc_id=account,
                        pr_node_id=node_id,
                        release_match=release_match,
                        repository_full_name=repo,
                        checked_until=checked_until,
                        merged_at=merged_at,
                        author=author,
                        merger=merger,
                        activity_days={},
                        labels={label: "" for label in labels.get(node_id, [])},
                    )
                    .create_defaults()
                    .explode(with_primary_keys=True),
                )
        if not values:
            unreleased_prs_event.set()
            return
        if postgres:
            sql = postgres_insert(GitHubMergedPullRequestFacts)
            sql = sql.on_conflict_do_update(
                constraint=GitHubMergedPullRequestFacts.__table__.primary_key,
                set_={
                    GitHubMergedPullRequestFacts.checked_until.name: greatest(
                        GitHubMergedPullRequestFacts.checked_until, sql.excluded.checked_until,
                    ),
                    GitHubMergedPullRequestFacts.labels.name: GitHubMergedPullRequestFacts.labels
                    + sql.excluded.labels,
                    GitHubMergedPullRequestFacts.updated_at.name: sql.excluded.updated_at,
                    GitHubMergedPullRequestFacts.data.name: GitHubMergedPullRequestFacts.data,
                },
            )
        else:
            # this is wrong but we just cannot update SQLite properly
            # nothing will break though
            sql = insert(GitHubMergedPullRequestFacts).prefix_with("OR REPLACE")
    try:
        with sentry_sdk.start_span(op="update_unreleased_prs/execute"):
            if pdb.url.dialect == "sqlite":
                async with pdb.connection() as pdb_conn:
                    async with pdb_conn.transaction():
                        await pdb_conn.execute_many(sql, values)
            else:
                # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
                await pdb.execute_many(sql, values)
    finally:
        unreleased_prs_event.set()


@sentry_span
async def store_merged_unreleased_pull_request_facts(
    merged_prs_and_facts: Iterable[Tuple[MinedPullRequest, PullRequestFacts]],
    matched_bys: Dict[str, ReleaseMatch],
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    account: int,
    pdb: morcilla.Database,
    unreleased_prs_event: asyncio.Event,
) -> None:
    """
    Persist the facts about merged unreleased pull requests to the database.

    Each passed PR must be merged and not released, we raise an assertion otherwise.
    """
    postgres = pdb.url.dialect == "postgresql"
    if not postgres:
        assert pdb.url.dialect == "sqlite"
    values = []
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for pr, facts in merged_prs_and_facts:
        assert facts.merged and not facts.released
        repo = pr.pr[PullRequest.repository_full_name.name]
        if (
            release_match := extract_release_match(
                repo, matched_bys, default_branches, release_settings,
            )
        ) is None:
            # no new releases
            continue
        values.append(
            GitHubMergedPullRequestFacts(
                acc_id=account,
                pr_node_id=pr.pr[PullRequest.node_id.name],
                release_match=release_match,
                repository_full_name=repo,
                data=facts.data,
                activity_days=collect_activity_days(pr, facts, not postgres),
                # the following does not matter, are not updated so we set to 0xdeadbeef
                checked_until=dt,
                merged_at=dt,
                author=None,
                merger=None,
                labels={},
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        )
    await unreleased_prs_event.wait()
    ghmprf = GitHubMergedPullRequestFacts
    if postgres:
        sql = postgres_insert(ghmprf)
        sql = sql.on_conflict_do_update(
            constraint=ghmprf.__table__.primary_key,
            set_={
                ghmprf.data.name: sql.excluded.data,
                ghmprf.activity_days.name: sql.excluded.activity_days,
            },
        )
        with sentry_sdk.start_span(op="store_merged_unreleased_pull_request_facts/execute"):
            if pdb.url.dialect == "sqlite":
                async with pdb.connection() as pdb_conn:
                    async with pdb_conn.transaction():
                        await pdb_conn.execute_many(sql, values)
            else:
                # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
                await pdb.execute_many(sql, values)
    else:
        tasks = [
            pdb.execute(
                update(ghmprf)
                .where(
                    and_(
                        ghmprf.pr_node_id == v[ghmprf.pr_node_id.name],
                        ghmprf.release_match == v[ghmprf.release_match.name],
                        ghmprf.format_version == v[ghmprf.format_version.name],
                    ),
                )
                .values(
                    {
                        ghmprf.data: v[ghmprf.data.name],
                        ghmprf.activity_days: v[ghmprf.activity_days.name],
                        ghmprf.updated_at: datetime.now(timezone.utc),
                    },
                ),
            )
            for v in values
        ]
        await gather(*tasks)


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, participants, labels, default_branches, release_settings, **_: (  # noqa
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repos)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        labels,
        sorted(default_branches.items()),
        release_settings,
    ),
    refresh_on_access=True,
)
async def discover_inactive_merged_unreleased_prs(
    time_from: datetime,
    time_to: datetime,
    repos: Union[Set[str], KeysView[str]],
    participants: PRParticipants,
    labels: LabelFilter,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    prefixer: Prefixer,
    account: int,
    pdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
) -> Dict[int, List[str]]:
    """
    Discover PRs which were merged before `time_from` and still not released.

    :return: mapping PR node ID -> logical repository names.
    """
    assert isinstance(repos, (set, KeysView))
    postgres = pdb.url.dialect == "postgresql"
    ghmprf = GitHubMergedPullRequestFacts
    ghdprf = GitHubDonePullRequestFacts
    selected = {
        ghmprf.pr_node_id,
        ghmprf.repository_full_name,
        ghmprf.release_match,
    }
    filters = [
        coalesce(ghdprf.pr_done_at, datetime(3000, 1, 1, tzinfo=timezone.utc)) >= time_to,
        ghmprf.repository_full_name.in_(repos),
        ghmprf.merged_at < time_from,
        ghmprf.acc_id == account,
    ]
    user_login_to_node_get = None
    for role, col in (
        (PRParticipationKind.AUTHOR, ghmprf.author),
        (PRParticipationKind.MERGER, ghmprf.merger),
    ):
        if people := participants.get(role):
            if user_login_to_node_get is None:
                user_login_to_node_get = prefixer.user_login_to_node.get
            filters.append(
                col.in_(list(chain.from_iterable(user_login_to_node_get(u, []) for u in people))),
            )
    if labels:
        build_labels_filters(ghmprf, labels, filters, selected, postgres)
    body = join(
        ghmprf,
        ghdprf,
        and_(
            ghdprf.acc_id == ghmprf.acc_id,
            ghdprf.pr_node_id == ghmprf.pr_node_id,
            ghdprf.release_match == ghmprf.release_match,
            ghdprf.pr_created_at < time_from,
        ),
        isouter=True,
    )
    selected = sorted(selected, key=lambda i: i.key)
    with sentry_sdk.start_span(op="load_inactive_merged_unreleased_prs/fetch"):
        rows = await pdb.fetch_all(
            select(selected)
            .select_from(body)
            .where(and_(*filters))
            .order_by(desc(GitHubMergedPullRequestFacts.merged_at)),
        )
    ambiguous = {ReleaseMatch.tag.name: set(), ReleaseMatch.branch.name: set()}
    node_ids = []
    repos = []
    if labels and not postgres:
        include_singles, include_multiples = LabelFilter.split(labels.include)
        include_singles = set(include_singles)
        include_multiples = [set(m) for m in include_multiples]
    result = {}
    remove_physical = set()
    for row in rows:
        dump = triage_by_release_match(
            row[ghmprf.repository_full_name.name],
            row[ghmprf.release_match.name],
            release_settings,
            default_branches,
            "whatever",
            ambiguous,
        )
        if dump is None:
            continue
        # we do not care about the exact release match
        if (
            labels
            and not postgres
            and not labels_are_compatible(
                include_singles,
                include_multiples,
                labels.exclude,
                row[GitHubMergedPullRequestFacts.labels.name],
            )
        ):
            continue
        node_id, repo = row[ghmprf.pr_node_id.name], row[ghmprf.repository_full_name.name]
        if (physical_repo := drop_logical_repo(repo)) != repo and physical_repo in repos:
            remove_physical.add((node_id, physical_repo))
        result.setdefault(node_id, []).append(repo)
    for node_id, repo in remove_physical:
        try:
            result[node_id].remove(repo)
        except ValueError:
            continue
    add_pdb_hits(pdb, "inactive_merged_unreleased", len(node_ids))
    return result
