from collections import defaultdict
from datetime import datetime, timedelta
from itertools import chain
import logging
from typing import Collection, Iterable, List, Optional, Tuple, Union
import warnings

import aiomcache
import medvedi as md
import numpy as np
from sqlalchemy import BigInteger, and_, exists, func, select, type_coerce, union_all
from xxhash import xxh3_64_intdigest

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query, read_sql_query_with_join_collapse
from athenian.api.cache import cached, middle_term_exptime, short_term_exptime
from athenian.api.db import Database, DatabaseLike
from athenian.api.int_to_str import int_to_str
from athenian.api.internal.features.github.check_run_metrics_accelerated import (
    mark_check_suite_types,
)
from athenian.api.internal.logical_repos import coerce_logical_repos, contains_logical_repos
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.check_run_accelerated import split_duplicate_check_runs
from athenian.api.internal.miners.github.label import (
    fetch_labels_to_filter,
    find_left_prs_by_labels,
)
from athenian.api.internal.miners.github.logical import split_logical_prs
from athenian.api.internal.miners.jira.issue import generate_jira_prs_query
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.metadata.github import (
    CheckRun,
    CheckRunByPR,
    NodePullRequest,
    NodePullRequestCommit,
    NodeRepository,
    PullRequestLabel,
)
from athenian.api.object_arrays import as_bool, objects_to_pyunicode_bytes
from athenian.api.tracing import sentry_span

maximum_processed_check_runs = 300_000
check_suite_started_column = "check_suite_started"
check_suite_completed_column = "check_suite_completed"
pull_request_started_column = "pull_request_" + NodePullRequest.created_at.name
pull_request_closed_column = "pull_request_" + NodePullRequest.closed_at.name
pull_request_merged_column = "pull_request_" + NodePullRequest.merged.name
pull_request_title_column = "pull_request_" + NodePullRequest.title.name


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=md.DataFrame.serialize_unsafe,
    deserialize=md.DataFrame.deserialize_unsafe,
    key=lambda time_from, time_to, repositories, pushers, labels, jira, logical_settings, **_: (  # noqa
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repositories)),
        ",".join(sorted(pushers)),
        labels,
        jira,
        logical_settings,
    ),
)
async def mine_check_runs(
    time_from: datetime,
    time_to: datetime,
    repositories: Collection[str],
    pushers: Collection[str],
    labels: LabelFilter,
    jira: JIRAFilter,
    logical_settings: LogicalRepositorySettings,
    meta_ids: Tuple[int, ...],
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> md.DataFrame:
    """
    Filter check runs according to the specified parameters.

    Relationship:

    Commit -> Check Suite -> one or more Check Runs.
    The same commit may appear in 0, 1, or more PRs. We have to disambiguate cases when there is
    more than one PR mapped to the same commit by looking at their lifetimes [created, closed].

    :param time_from: Check runs must start later than this time.
    :param time_to: Check runs must start earlier than this time.
    :param repositories: Look for check runs in these repository names (without prefix).
    :param pushers: Check runs must link to the commits with the given pusher logins.
    :param labels: Check runs must link to PRs marked with these labels.
    :param jira: Check runs must link to PRs satisfying this JIRA filter.
    :return: Pandas DataFrame with columns mapped from CheckRun model.
    """
    log = logging.getLogger(f"{metadata.__package__}.mine_check_runs")
    assert time_from.tzinfo is not None
    assert time_to.tzinfo is not None
    coerced_repositories = coerce_logical_repos(repositories)
    with_logical_repos = contains_logical_repos(repositories)
    rows = await mdb.fetch_all(
        select(NodeRepository.acc_id, NodeRepository.id).where(
            NodeRepository.acc_id.in_(meta_ids),
            NodeRepository.name_with_owner.in_(coerced_repositories),
        ),
    )
    if len(meta_ids) == 1:
        repo_nodes = {meta_ids[0]: [r[NodeRepository.id.name] for r in rows]}
    else:
        repo_nodes = defaultdict(list)
        for row in rows:
            repo_nodes[row[NodeRepository.acc_id.name]].append(row[NodeRepository.id.name])
    del rows

    queries = []
    _common_filters = []

    def common_filters():
        nonlocal _common_filters
        lookaround = time_from - timedelta(days=14), time_to + timedelta(days=1)
        if not _common_filters or mdb.url.dialect == "sqlite":
            _common_filters = [
                CheckRun.started_at.between(time_from, time_to),  # original seed
                CheckRun.committed_date_hack.between(*lookaround),  # HashJoin constraint
                CheckRun.committed_date.between(*lookaround),  # HashJoin constraint
            ]
            if len(pushers):
                _common_filters.append(CheckRun.author_login.in_(pushers))
        return _common_filters

    for acc_id, acc_repo_nodes in repo_nodes.items():
        # our usual acc_id.in_(meta_ids) breaks the planner here and we explode
        filters = [
            CheckRun.acc_id == acc_id,
            CheckRun.repository_node_id.in_(acc_repo_nodes),
            *common_filters(),
        ]
        if labels:
            singles, multiples = LabelFilter.split(labels.include)
            embedded_labels_query = not multiples and not labels.exclude
            if not labels.exclude:
                all_in_labels = set(singles + list(chain.from_iterable(multiples)))
                filters.append(
                    exists().where(
                        PullRequestLabel.acc_id == CheckRun.acc_id,
                        PullRequestLabel.pull_request_node_id == CheckRun.pull_request_node_id,
                        func.lower(PullRequestLabel.name).in_(all_in_labels),
                    ),
                )
        else:
            embedded_labels_query = False
        if not jira:
            query = select(CheckRun).where(and_(*filters))
        else:
            query = await generate_jira_prs_query(
                filters,
                jira,
                None,
                mdb,
                cache,
                columns=CheckRun.__table__.columns,
                seed=CheckRun,
                on=(CheckRun.pull_request_node_id, CheckRun.acc_id),
            )
        query = (
            query.with_statement_hint("IndexOnlyScan(c_1 github_node_commit_check_runs)")
            .with_statement_hint("IndexOnlyScan(p github_node_push_redux)")
            .with_statement_hint("IndexOnlyScan(prc node_pull_request_commit_commit_pr)")
            .with_statement_hint("IndexScan(pr node_pullrequest_pkey)")
            .with_statement_hint("IndexScan(sc ath_node_statuscontext_commit_created_at)")
            .with_statement_hint("IndexScan(cr github_node_check_run_repository_started_at)")
            .with_statement_hint("Rows(cr cs *400)")
            .with_statement_hint("Rows(cr cs c *400)")
            .with_statement_hint("Rows(c_1 sc *1000)")
            .with_statement_hint("HashJoin(cr cs)")
            .with_statement_hint("Set(enable_parallel_append 0)")
        )
        """
        PostgreSQL has no idea about column correlations between tables, and extended statistics
        only helps with correlations within the same table. That's the ultimate problem that leads
        to very poor plans.
        1. We enforce the best JOIN order. Vadim has spent *much* time figuring it out across different
        accounts.
        2. Adjust the number of rows in the core INNER JOIN-s of both UNION branches. This decreases
        the probability of blowing up on NESTED LOOP-s where we should have MERGE JOIN or HASH JOIN.
        The exact multipliers are a compromise between a few accounts Vadim tested.
        3. Still, Postgres sucks at choosing the right indexes sometimes. We pin the critical ones.
        Yet some indexes shouldn't be pinned because of different plans on different accounts
        (different balance between UNION branches). Particularly, Vadim tried to mess with `cs`and
        failed: https://github.com/athenianco/athenian-api/commit/4038e75bdd66ab80c4ba0e561ac48c5b71f797f8#diff-2cc6b19d09c47c95d39a3fdf03116425827c3fc942b9be97f66f59722f9430bb
        It helped a little with one account but completely destroyed the others.
        4. We disable PARALLEL APPEND. Whatever Vadim tried, Postgres always schedules only one worker,
        effectively executing UNION branches sequentially.
        5. We enforce the hash join between cr and cs, no matter the number of rows. This is required
        to let committed_date_hack pre-filter check suites. Otherwise, it becomes a considerable overhead
        on bigger row counts.
        """  # noqa: E501
        queries.append(query)

    query = (
        queries[0].limit(maximum_processed_check_runs)
        if len(queries) == 1
        else union_all(*queries)
    )
    df = await read_sql_query_with_join_collapse(
        query, mdb, CheckRun, soft_limit=maximum_processed_check_runs,
    )

    # add check runs mapped to the mentioned PRs even if they are outside of the date range
    df, df_labels = await _append_pull_request_check_runs_outside(
        df, time_from, time_to, labels, embedded_labels_query, meta_ids, mdb,
    )

    # the same check runs / suites may attach to different PRs, fix that
    df, *pr_labels = await _disambiguate_pull_requests(df, with_logical_repos, log, meta_ids, mdb)

    # deferred filter by labels so that we disambiguate PRs always the same way
    if labels:
        df.take(df[CheckRun.pull_request_node_id.name].astype(bool), inplace=True)
        df.reset_index(inplace=True)
        if not embedded_labels_query:
            df = _filter_by_pr_labels(df, labels, df_labels)

    df = _finalize_check_runs(df, log)

    if with_logical_repos:
        df = split_logical_prs(
            df,
            *pr_labels,
            repositories,
            logical_settings,
            reindex=False,
            reset_index=False,
            repo_column=CheckRun.repository_full_name.name,
            id_column=CheckRun.pull_request_node_id.name,
            title_column=pull_request_title_column,
        )
        df.reset_index(inplace=True)

    return df


def _finalize_check_runs(df: md.DataFrame, log: logging.Logger) -> md.DataFrame:
    # some status contexts represent the start and the finish events, join them together
    df_len = len(df)
    df = _merge_status_contexts(df)
    log.info("merged %d / %d", df_len - len(df), df_len)

    # "Re-run jobs" may produce duplicate check runs in the same check suite, split them
    # in separate artificial check suites by enumerating in chronological order
    split = _split_duplicate_check_runs(df)
    log.info("split %d / %d", split, len(df))

    _postprocess_check_runs(df)

    return df


@sentry_span
async def _disambiguate_pull_requests(
    df: md.DataFrame,
    with_logical_repo_support: bool,
    log: logging.Logger,
    meta_ids: Tuple[int, ...],
    mdb: Database,
) -> Union[Tuple[md.DataFrame], Tuple[md.DataFrame, md.DataFrame]]:
    pr_node_ids = df[CheckRun.pull_request_node_id.name]
    check_run_node_ids = df[CheckRun.check_run_node_id.name]
    unique_node_ids, node_id_counts = np.unique(check_run_node_ids, return_counts=True)
    ambiguous_unique_check_run_indexes = np.flatnonzero(node_id_counts > 1)
    if len(ambiguous_unique_check_run_indexes):
        log.debug("must disambiguate %d check runs", len(ambiguous_unique_check_run_indexes))
        # another groupby() replacement for speed
        # we determine check runs belonging to the same commit with multiple pull requests
        node_ids_order = np.argsort(check_run_node_ids)
        stops = np.cumsum(node_id_counts)[ambiguous_unique_check_run_indexes]
        lengths = node_id_counts[ambiguous_unique_check_run_indexes]
        groups = np.repeat(stops - lengths.cumsum(), lengths) + np.arange(lengths.sum())
        unique_ambiguous_pr_node_ids = np.unique(pr_node_ids[node_ids_order[groups]])
    else:
        unique_ambiguous_pr_node_ids = []

    # we need all PR lifetimes to check explicit node_id indexes
    pr_cols = [NodePullRequest.id, NodePullRequest.author_id, NodePullRequest.merged]
    if fetch_pr_ts := pull_request_started_column not in df.columns:
        # coming from _append_pull_request_check_runs_outside() / mine_check_runs()
        assert pull_request_closed_column not in df.columns
        pr_cols.extend([NodePullRequest.created_at, NodePullRequest.closed_at])
    else:
        # coming from mine_commit_check_runs()
        assert pull_request_closed_column in df.columns
    if with_logical_repo_support:
        pr_cols.append(NodePullRequest.title)
    unique_pr_ids = np.unique(pr_node_ids)  # with 0, but that's fine
    pr_lifetimes, pr_commit_counts, *pr_labels = await gather(
        read_sql_query(
            select(*pr_cols).where(
                NodePullRequest.acc_id.in_(meta_ids),
                NodePullRequest.id.in_any_values(unique_pr_ids),
            ),
            mdb,
            pr_cols,
            index=NodePullRequest.id.name,
        ),
        read_sql_query(
            select(
                *(
                    pr_commit_counts_cols := [
                        NodePullRequestCommit.pull_request_id,
                        type_coerce(func.count(NodePullRequestCommit.commit_id), BigInteger).label(
                            "count",
                        ),
                    ]
                ),
            )
            .where(
                NodePullRequestCommit.acc_id.in_(meta_ids),
                NodePullRequestCommit.pull_request_id.in_any_values(unique_ambiguous_pr_node_ids),
            )
            .group_by(NodePullRequestCommit.pull_request_id),
            mdb,
            pr_commit_counts_cols,
            index=NodePullRequestCommit.pull_request_id.name,
        ),
        *(
            [fetch_labels_to_filter(unique_pr_ids, meta_ids, mdb)]
            if with_logical_repo_support
            else []
        ),
    )
    del unique_ambiguous_pr_node_ids
    pr_lifetimes.rename(
        columns={
            **(
                {
                    NodePullRequest.created_at.name: pull_request_started_column,
                    NodePullRequest.closed_at.name: pull_request_closed_column,
                }
                if fetch_pr_ts
                else {}
            ),
            NodePullRequest.merged.name: pull_request_merged_column,
            **(
                {NodePullRequest.title.name: pull_request_title_column}
                if with_logical_repo_support
                else {}
            ),
            NodePullRequest.id.name: CheckRun.pull_request_node_id.name,
        },
        inplace=True,
    )
    df = md.join(
        df.set_index(CheckRun.pull_request_node_id.name, inplace=True),
        pr_lifetimes[
            [pull_request_merged_column]
            + ([pull_request_started_column, pull_request_closed_column] if fetch_pr_ts else [])
            + ([pull_request_title_column] if with_logical_repo_support else [])
        ],
    )
    df.fillna(np.datetime64(datetime.now(), "us"), pull_request_closed_column, inplace=True)
    df[pull_request_closed_column][df.isnull(pull_request_started_column)] = None
    df[pull_request_merged_column] = as_bool(df[pull_request_merged_column])
    if with_logical_repo_support:
        df.fillna("", pull_request_title_column, inplace=True)

    # do not let different check runs belonging to the same suite map to different PRs
    _calculate_check_suite_started(df)
    try:
        check_runs_outside_pr_lifetime_indexes = np.flatnonzero(
            (df[check_suite_started_column] < df[pull_request_started_column])
            | (
                df[check_suite_started_column]
                >= df[pull_request_closed_column] + np.timedelta64(1, "h")
            ),
        )
    except TypeError:
        # "Cannot compare tz-naive and tz-aware datetime-like objects"
        # all the timestamps are NAT-s
        check_runs_outside_pr_lifetime_indexes = np.arange(len(df))
    # check run must launch while the PR remains open
    df[CheckRun.pull_request_node_id.name][check_runs_outside_pr_lifetime_indexes] = 0
    old_df_len = len(df)
    df.drop_duplicates(
        [CheckRun.check_run_node_id.name, CheckRun.pull_request_node_id.name],
        inplace=True,
        ignore_index=True,
    )
    check_run_node_ids = df[CheckRun.check_run_node_id.name]
    pr_node_ids = df[CheckRun.pull_request_node_id.name].copy()
    check_suite_node_ids = df[CheckRun.check_suite_node_id.name]
    author_node_ids = df[CheckRun.author_user_id.name]
    pull_request_starteds = df[pull_request_started_column]
    log.info("rejecting check runs by PR lifetimes: %d / %d", len(df), old_df_len)

    if len(ambiguous_unique_check_run_indexes):
        # second lap
        unique_node_ids, node_id_counts = np.unique(check_run_node_ids, return_counts=True)
        ambiguous_unique_check_run_indexes = np.flatnonzero(node_id_counts > 1)
        if len(ambiguous_unique_check_run_indexes):
            node_ids_order = np.argsort(check_run_node_ids)
            stops = np.cumsum(node_id_counts)[ambiguous_unique_check_run_indexes]
            lengths = node_id_counts[ambiguous_unique_check_run_indexes]
            groups = np.repeat(stops - lengths.cumsum(), lengths) + np.arange(lengths.sum())
            ambiguous_indexes = node_ids_order[groups]
            log.info("must disambiguate %d check runs", len(ambiguous_indexes))
            ambiguous_pr_node_ids = pr_node_ids[ambiguous_indexes]
    if len(ambiguous_unique_check_run_indexes):
        ambiguous_check_run_node_ids = check_run_node_ids[ambiguous_indexes]
        ambiguous_df = md.join(
            md.DataFrame(
                {
                    CheckRun.check_run_node_id.name: ambiguous_check_run_node_ids,
                    CheckRun.check_suite_node_id.name: check_suite_node_ids[ambiguous_indexes],
                    CheckRun.pull_request_node_id.name: ambiguous_pr_node_ids,
                    CheckRun.author_user_id.name: author_node_ids[ambiguous_indexes],
                    pull_request_started_column: pull_request_starteds[ambiguous_indexes],
                    "ambiguous_indexes": ambiguous_indexes,
                },
                index=CheckRun.pull_request_node_id.name,
            ),
            pr_lifetimes[[NodePullRequest.author_id.name]],
        )
        # we need to sort to stabilize idxmin() in step 2
        ambiguous_df.sort_values(pull_request_started_column, inplace=True)
        # heuristic: the PR should be created by the commit author
        passed = (
            ambiguous_df[NodePullRequest.author_id.name]
            == ambiguous_df[CheckRun.author_user_id.name]
        )
        log.info("disambiguation step 1 - authors: %d / %d", passed.sum(), len(ambiguous_df))
        passed_df = md.join(ambiguous_df.take(passed, inplace=True), pr_commit_counts)
        del ambiguous_df
        # heuristic: the PR with the least number of commits wins
        order = np.argsort(passed_df["count"], kind="stable")
        _, first_encounters = np.unique(
            passed_df[CheckRun.check_run_node_id.name][order], return_index=True,
        )
        passed = order[first_encounters]
        log.info("disambiguation step 2 - commit counts: %d / %d", len(passed), len(passed_df))
        # we may discard some check runs completely here, set pull_request_node_id to None for them
        reset_indexes = np.setdiff1d(
            ambiguous_indexes, passed_df["ambiguous_indexes"][passed], assume_unique=True,
        )
        del passed_df
        log.info("disambiguated null-s: %d / %d", len(reset_indexes), len(ambiguous_indexes))
        df[CheckRun.pull_request_node_id.name][reset_indexes] = 0
        # there can be check runs mapped to both a PR and None; remove None-s
        pr_node_ids[reset_indexes] = -1
        pr_node_ids[pr_node_ids == 0] = -1
        joint = int_to_str(check_run_node_ids, pr_node_ids)
        order = np.argsort(joint)
        _, first_encounters = np.unique(check_run_node_ids[order], return_index=True)
        first_encounters = order[first_encounters]
        # first_encounters either map to a PR or to the only None for each check run
        log.info("final size: %d / %d", len(first_encounters), len(df))
        df.take(first_encounters, inplace=True)
        df.reset_index(inplace=True)

    return df, *pr_labels


@sentry_span
def _erase_completed_at_as_needed(df: md.DataFrame):
    if df.empty:
        return

    # exclude skipped checks from execution time calculation
    df[CheckRun.completed_at.name][df[CheckRun.conclusion.name] == b"NEUTRAL"] = None

    # exclude "too quick" checks from execution time calculation DEV-3155
    cr_types = np.empty(len(df), dtype=[("repo", int), ("name", np.uint64)])
    cr_types["repo"] = df[CheckRun.repository_node_id.name].byteswap()
    cr_names = df[CheckRun.name.name].astype("U", copy=False)
    # take the hash to avoid very long names - they lead to bad performance
    cr_types["name"] = np.fromiter(
        (xxh3_64_intdigest(s) for s in cr_names.view(f"S{cr_names.dtype.itemsize}")),
        np.uint64,
        len(df),
    )
    cr_types = cr_types.view("S16")  # speeds up np.unique()
    first_encounters, suite_groups = mark_check_suite_types(
        cr_types, df[CheckRun.check_suite_node_id.name],
    )
    del cr_types
    _, cs_type_counts = np.unique(suite_groups, return_counts=True)
    max_cs_count = cs_type_counts.max()
    cs_type_offsets = cs_type_counts.cumsum()
    cs_indexes = np.repeat(
        np.arange(len(cs_type_counts)) * max_cs_count + cs_type_counts - cs_type_offsets,
        cs_type_counts,
    ) + np.arange(len(suite_groups))
    cs_run_times_shaped = np.full((len(cs_type_counts), max_cs_count), None, "timedelta64[s]")
    cs_run_times_shaped.ravel()[cs_indexes] = (
        df[check_suite_completed_column] - df[check_suite_started_column]
    )[first_encounters[np.argsort(suite_groups)]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        median_cs_run_times = np.nanmedian(cs_run_times_shaped, axis=-1)
    excluded_cs_types = np.flatnonzero(median_cs_run_times <= np.timedelta64(8, "s"))
    excluded_cs = df[CheckRun.check_suite_node_id.name][
        first_encounters[np.in1d(suite_groups, excluded_cs_types)]
    ]
    excluded_mask = df.isin(CheckRun.check_suite_node_id.name, excluded_cs)
    df[check_suite_completed_column][excluded_mask] = None


@sentry_span
def _postprocess_check_runs(df: md.DataFrame) -> None:
    # there can be checks that finished before starting 🤦‍
    # md.DataFrame.max(axis=1) does not work correctly because of the NaT-s
    started_ats = df[CheckRun.started_at.name]
    df[CheckRun.completed_at.name] = np.maximum(df[CheckRun.completed_at.name], started_ats)
    df[CheckRun.completed_at.name] = df[CheckRun.completed_at.name].astype(
        started_ats.dtype, copy=False,
    )
    grouper = df.groupby(CheckRun.check_suite_node_id.name)
    df[check_suite_completed_column] = values = np.empty_like(
        completed_ats := df[CheckRun.completed_at.name],
    )
    values[grouper.order] = np.repeat(
        np.fmax.reduceat(completed_ats[grouper.order], grouper.reduceat_indexes()),
        grouper.counts,
    )

    _erase_completed_at_as_needed(df)

    # ensure that the timestamps are in sync after pruning PRs
    pr_ts_columns = [
        pull_request_started_column,
        pull_request_closed_column,
    ]
    zero_mask = df[CheckRun.pull_request_node_id.name] == 0
    for column in pr_ts_columns:
        assert df[column].dtype == df[CheckRun.started_at.name].dtype
        df[column][zero_mask] = np.datetime64("NaT")
    if pull_request_title_column in df:
        df[pull_request_title_column][zero_mask] = ""
    df[pull_request_merged_column][zero_mask] = False

    for col in (
        CheckRun.check_run_node_id,
        CheckRun.check_suite_node_id,
        CheckRun.repository_node_id,
        CheckRun.commit_node_id,
    ):
        assert df[col.name].dtype == int, col.name


@sentry_span
async def _append_pull_request_check_runs_outside(
    df: md.DataFrame,
    time_from: datetime,
    time_to: datetime,
    labels: LabelFilter,
    embedded_labels_query: bool,
    meta_ids: Tuple[int, ...],
    mdb: Database,
) -> [md.DataFrame, Tuple[md.DataFrame, ...]]:
    pr_node_ids = df[CheckRun.pull_request_node_id.name]
    prs_from_the_past = np.unique(
        pr_node_ids[
            df[CheckRun.pull_request_created_at.name]
            < np.datetime64((time_from + timedelta(days=1)).replace(tzinfo=None), "us")
        ],
    )
    prs_to_the_future = np.unique(
        pr_node_ids[
            df.isnull(CheckRun.pull_request_closed_at.name)
            | (
                df[CheckRun.pull_request_closed_at.name]
                > np.datetime64((time_to - timedelta(days=1)).replace(tzinfo=None), "us")
            )
        ],
    )
    query_before = select(CheckRunByPR).where(
        CheckRunByPR.acc_id.in_(meta_ids),
        CheckRunByPR.pull_request_node_id.in_(prs_from_the_past),
        CheckRunByPR.started_at.between(
            time_from - timedelta(days=90), time_from - timedelta(seconds=1),
        ),
    )
    query_after = select(CheckRunByPR).where(
        CheckRunByPR.acc_id.in_(meta_ids),
        CheckRunByPR.pull_request_node_id.in_(prs_to_the_future),
        CheckRunByPR.started_at.between(
            time_to + timedelta(seconds=1), time_to + timedelta(days=90),
        ),
    )
    pr_sql = union_all(query_before, query_after)
    tasks = [
        read_sql_query_with_join_collapse(pr_sql, mdb, CheckRunByPR),
    ]
    if labels and not embedded_labels_query:
        tasks.append(
            read_sql_query(
                select(
                    PullRequestLabel.pull_request_node_id,
                    func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name),
                ).where(
                    PullRequestLabel.acc_id.in_(meta_ids),
                    PullRequestLabel.pull_request_node_id.in_(np.unique(pr_node_ids)),
                ),
                mdb,
                [PullRequestLabel.pull_request_node_id, PullRequestLabel.name],
                index=PullRequestLabel.pull_request_node_id.name,
            ),
        )
    extra_df, *df_labels = await gather(*tasks)
    for col in (
        CheckRun.committed_date_hack,
        CheckRun.pull_request_created_at,
        CheckRun.pull_request_closed_at,
    ):
        del df[col.name]
    if not extra_df.empty:
        df = md.concat(df, extra_df, ignore_index=True)
    df[CheckRun.completed_at.name] = df[CheckRun.completed_at.name].astype(
        df[CheckRun.started_at.name].dtype,
    )
    return df, df_labels


def _calculate_check_suite_started(df: md.DataFrame) -> None:
    grouper = df.groupby(CheckRun.check_suite_node_id.name)
    df[check_suite_started_column] = values = np.empty_like(
        started_ats := df[CheckRun.started_at.name],
    )
    values[grouper.order] = np.repeat(
        np.minimum.reduceat(started_ats[grouper.order], grouper.reduceat_indexes()),
        grouper.counts,
    )


@sentry_span
def _filter_by_pr_labels(
    df: md.DataFrame,
    labels: LabelFilter,
    df_labels: Tuple[md.DataFrame, ...],
) -> md.DataFrame:
    df_labels = df_labels[0]
    prs_left = find_left_prs_by_labels(
        df[CheckRun.pull_request_node_id.name],
        df_labels.index.values,
        df_labels[PullRequestLabel.name.name],
        labels,
    )
    mask_left = df.isin(CheckRun.pull_request_node_id.name, prs_left)
    if not mask_left.all():
        df.take(mask_left, inplace=True)
        df.reset_index(drop=True)
    return df


@sentry_span
def _merge_status_contexts(df: md.DataFrame) -> md.DataFrame:
    """
    Infer the check run completion time from two separate events: PENDING + SUCCESS/FAILURE/etc.

    See: DEV-2610, DEV-3103.
    """
    if df.empty:
        return df
    # There are 4 possible status context statuses:
    # ERROR
    # FAILURE
    # SUCCESS
    # PENDING
    #
    # Required order:
    # PENDING, ERROR, FAILURE, SUCCESS
    pending_repl = b"AAAAAAA"
    statuses = df[CheckRun.status.name].copy()
    statuses[statuses == b"PENDING"] = pending_repl
    starteds = df[CheckRun.started_at.name]
    # we *must* sort by the these:
    # 1. check run start time - general sequence
    # 2. check run status, PENDING must be the first - for merges
    # 3. check run name - for splits
    names32 = objects_to_pyunicode_bytes(df[CheckRun.name.name], 128)
    order = np.argsort(np.char.add(np.char.add(int_to_str(starteds.view(int)), statuses), names32))
    df.take(order, inplace=True)
    df.reset_index(inplace=True)
    statuses = statuses[order]
    starteds = starteds[order]
    del order
    # Important: we must sort in any case
    no_finish = np.flatnonzero(df.isnull(CheckRun.completed_at.name))
    if len(no_finish) == 0:
        # sweet dreams...
        return df
    no_finish_urls = df[CheckRun.url.name][no_finish].astype("S")
    empty_url_mask = no_finish_urls == b""
    empty_url_names = df[CheckRun.name.name][no_finish[empty_url_mask]].astype("U")
    no_finish_urls[empty_url_mask] = empty_url_names.view(f"S{empty_url_names.dtype.itemsize}")
    no_finish_parents = df[CheckRun.check_suite_node_id.name][no_finish].astype(int, copy=False)
    no_finish_seeds = np.char.add(int_to_str(no_finish_parents), no_finish_urls)
    _, indexes, counts = np.unique(no_finish_seeds, return_inverse=True, return_counts=True)
    firsts = np.zeros(len(counts), dtype=int)
    np.cumsum(counts[:-1], out=firsts[1:])
    lasts = np.roll(firsts, -1)
    lasts[-1] = len(indexes)
    lasts -= 1
    # order by CheckRun.started_at + (PENDING, ERROR, FAILURE, SUCCESS) + check suite + run type
    indexes_original = np.argsort(indexes, kind="stable")
    no_finish_original = no_finish[indexes_original]
    no_finish_original_statuses = statuses[no_finish_original]
    matched_beg = no_finish_original_statuses[firsts] == pending_repl  # replaced b"PENDING"
    matched_end = counts > 1  # the status does not matter
    matched = matched_beg & matched_end
    matched_firsts = firsts[matched]
    matched_lasts = lasts[matched]

    # calculate the indexes of removed (merged) check runs
    drop_mask = np.zeros(len(no_finish_original), bool)
    lengths = matched_lasts - matched_firsts
    dropped = np.repeat(matched_lasts - lengths.cumsum(), lengths) + np.arange(lengths.sum())
    drop_mask[dropped] = True
    dropped = no_finish_original[drop_mask]

    matched_first_starteds = starteds[no_finish_original[matched_firsts]]
    matched_indexes = no_finish_original[matched_lasts]
    df[CheckRun.started_at.name][matched_indexes] = matched_first_starteds
    completed_ats = starteds[matched_indexes]
    # DEV-4001
    completed_ats[(completed_ats - matched_first_starteds) > np.timedelta64(24, "h")] = None
    df[CheckRun.completed_at.name][matched_indexes] = completed_ats
    if len(dropped):
        leave_mask = np.ones(len(df), dtype=bool)
        leave_mask[dropped] = False
        df.take(leave_mask, inplace=True)
        df.reset_index(inplace=True)
    return df


@sentry_span
def _split_duplicate_check_runs(df: md.DataFrame) -> int:
    # DEV-2612 split older re-runs to artificial check suites
    # we require the df to be sorted by CheckRun.started_at
    if df.empty:
        return 0
    check_suite_node_ids = df[CheckRun.check_suite_node_id.name]
    split = split_duplicate_check_runs(
        check_suite_node_ids,
        df[CheckRun.name.name],
        df[CheckRun.started_at.name].astype("datetime64[s]"),
    )
    if split == 0:
        return 0
    check_run_conclusions = df[CheckRun.conclusion.name]
    check_suite_conclusions = df[CheckRun.check_suite_conclusion.name]
    successful = (check_suite_conclusions == b"SUCCESS") | (check_suite_conclusions == b"NEUTRAL")
    # override the successful conclusion of the check suite if at least one check run's conclusion
    # does not agree
    for c in (b"TIMED_OUT", b"CANCELLED", b"FAILURE"):  # the order matters
        mask = successful & np.in1d(
            check_suite_node_ids,
            np.unique(check_suite_node_ids[check_run_conclusions == c]),
        )
        if mask.any():
            df[CheckRun.check_suite_conclusion.name][mask] = c
    _calculate_check_suite_started(df)
    return split


@cached(
    exptime=middle_term_exptime,
    serialize=md.DataFrame.serialize_unsafe,
    deserialize=md.DataFrame.deserialize_unsafe,
    key=lambda commit_ids, **_: (",".join(map(str, sorted(commit_ids))),),
)
async def mine_commit_check_runs(
    commit_ids: Iterable[int],
    meta_ids: Tuple[int, ...],
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> md.DataFrame:
    """Fetch check runs belonging to the specified commits."""
    log = logging.getLogger(f"{metadata.__package__}.mine_commit_check_runs")
    dfs = await gather(
        *(
            read_sql_query(
                # IndexScan(c node_commit_pkey) -> DEV-3667
                select(CheckRun)
                .where(
                    CheckRun.acc_id == acc_id,
                    CheckRun.commit_node_id.in_(commit_ids),
                )
                .with_statement_hint("IndexOnlyScan(p github_node_push_redux)")
                .with_statement_hint("IndexOnlyScan(prc node_pull_request_commit_commit_pr)")
                .with_statement_hint("IndexScan(pr node_pullrequest_pkey)")
                .with_statement_hint("IndexScan(c node_commit_pkey)")
                .with_statement_hint("Rows(cr c *100)")
                .with_statement_hint("Rows(cr cs *100)")
                .with_statement_hint("Rows(cr cs c *2000)")
                .with_statement_hint("Rows(c_1 sc *20)")
                .with_statement_hint("Set(enable_parallel_append 0)"),
                mdb,
                CheckRun,
            )
            for acc_id in meta_ids
        ),
    )
    df = dfs[0]
    dfs = [df for df in dfs if not df.empty]
    if len(dfs) == 1:
        df = dfs[0]
    elif len(dfs) > 1:
        df = md.concat(dfs, ignore_index=True, copy=False)
    df[CheckRun.completed_at.name] = df[CheckRun.completed_at.name].astype(
        df[CheckRun.started_at.name].dtype,
    )
    df, *_ = await _disambiguate_pull_requests(df, False, log, meta_ids, mdb)
    return _finalize_check_runs(df, log)


def calculate_check_run_outcome_masks(
    check_run_statuses: np.ndarray,
    check_run_conclusions: np.ndarray,
    check_suite_conclusions: Optional[np.ndarray],
    with_success: bool,
    with_failure: bool,
    with_skipped: bool,
) -> List[np.ndarray]:
    """Calculate the check run success and failure masks."""
    completed = check_run_statuses == b"COMPLETED"
    if with_success or with_skipped:
        neutrals = check_run_conclusions == b"NEUTRAL"
    result = []
    if with_success:
        result.append(
            (
                completed
                & (
                    (check_run_conclusions == b"SUCCESS")
                    | (check_suite_conclusions == b"NEUTRAL") & neutrals
                )
            )
            | (check_run_statuses == b"SUCCESS")
            | (check_run_statuses == b"PENDING"),  # noqa(C812)
        )
    if with_failure:
        result.append(
            (
                completed
                & np.in1d(check_run_conclusions, [b"FAILURE", b"STALE", b"ACTION_REQUIRED"])
            )
            | (check_run_statuses == b"FAILURE")
            | (check_run_statuses == b"ERROR"),  # noqa(C812)
        )
    if with_skipped:
        result.append((check_suite_conclusions != b"NEUTRAL") & neutrals)
    return result
