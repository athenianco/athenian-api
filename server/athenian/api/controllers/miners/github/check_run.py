from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
from typing import Collection, Optional, Tuple, Union
import warnings

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, exists, func, select, union_all
from xxhash import xxh3_64_intdigest

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query, read_sql_query_with_join_collapse
from athenian.api.cache import cached, CancelCache, short_term_exptime
from athenian.api.controllers.features.github.check_run_metrics_accelerated import \
    mark_check_suite_types
from athenian.api.controllers.logical_repos import coerce_logical_repos, contains_logical_repos
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.label import fetch_labels_to_filter
from athenian.api.controllers.miners.github.logical import split_logical_repositories
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.settings import LogicalRepositorySettings
from athenian.api.db import Database, DatabaseLike
from athenian.api.int_to_str import int_to_str
from athenian.api.models.metadata.github import CheckRun, CheckRunByPR, NodePullRequest, \
    NodePullRequestCommit, NodeRepository, PullRequestLabel
from athenian.api.tracing import sentry_span


maximum_processed_check_runs = 300_000
check_suite_started_column = "check_suite_started"
check_suite_completed_column = "check_suite_completed"
pull_request_started_column = "pull_request_" + NodePullRequest.created_at.name
pull_request_closed_column = "pull_request_" + NodePullRequest.closed_at.name
pull_request_merged_column = "pull_request_" + NodePullRequest.merged.name
pull_request_title_column = "pull_request_" + NodePullRequest.title.name


async def mine_check_runs(time_from: datetime,
                          time_to: datetime,
                          repositories: Collection[str],
                          pushers: Collection[str],
                          labels: LabelFilter,
                          jira: JIRAFilter,
                          only_prs: bool,
                          logical_settings: LogicalRepositorySettings,
                          meta_ids: Tuple[int, ...],
                          mdb: DatabaseLike,
                          cache: Optional[aiomcache.Client],
                          ) -> pd.DataFrame:
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
    df, _ = await _mine_check_runs(
        time_from, time_to, repositories, pushers, labels, jira, only_prs, logical_settings,
        meta_ids, mdb, cache)
    return df


def _postprocess_only_prs(result: Tuple[pd.DataFrame, bool],
                          only_prs: bool,
                          **_) -> Tuple[pd.DataFrame, bool]:
    df, cached_only_prs = result
    if cached_only_prs and not only_prs:
        raise CancelCache()
    if only_prs:
        df = df.take(np.flatnonzero(df[CheckRun.pull_request_node_id.name].values != 0))
        df.reset_index(inplace=True, drop=True)
    return df, only_prs


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, pushers, labels, jira, logical_settings, **_:  # noqa
    (
        time_from.timestamp(), time_to.timestamp(),
        ",".join(sorted(repositories)),
        ",".join(sorted(pushers)),
        labels,
        jira,
        logical_settings,
    ),
    postprocess=_postprocess_only_prs,
)
async def _mine_check_runs(time_from: datetime,
                           time_to: datetime,
                           repositories: Collection[str],
                           pushers: Collection[str],
                           labels: LabelFilter,
                           jira: JIRAFilter,
                           only_prs: bool,
                           logical_settings: LogicalRepositorySettings,
                           meta_ids: Tuple[int, ...],
                           mdb: DatabaseLike,
                           cache: Optional[aiomcache.Client],
                           ) -> Tuple[pd.DataFrame, bool]:
    log = logging.getLogger(f"{metadata.__package__}.mine_check_runs")
    assert time_from.tzinfo is not None
    assert time_to.tzinfo is not None
    coerced_repositories = coerce_logical_repos(repositories)
    with_logical_repos = contains_logical_repos(repositories)
    repo_nodes = [r[0] for r in await mdb.fetch_all(
        select([NodeRepository.id])
        .where(and_(NodeRepository.acc_id.in_(meta_ids),
                    NodeRepository.name_with_owner.in_(coerced_repositories))))]
    filters = [
        CheckRun.acc_id.in_(meta_ids),
        CheckRun.started_at.between(time_from, time_to),
        CheckRun.committed_date_hack.between(
            time_from - timedelta(days=14), time_to + timedelta(days=1)),
        CheckRun.repository_node_id.in_(repo_nodes),
    ]
    # Postgres planner sucks at guessing the number of records here => terrible perf
    # if only_prs:
    #     filters.append(CheckRun.pull_request_node_id.isnot(None))
    # TODO(vmarkovtsev): hints should mitigate this problem
    if pushers:
        filters.append(CheckRun.author_login.in_(pushers))
    if labels:
        singles, multiples = LabelFilter.split(labels.include)
        embedded_labels_query = not multiples and not labels.exclude
        if not labels.exclude:
            all_in_labels = set(singles + list(chain.from_iterable(multiples)))
            filters.append(
                exists().where(and_(
                    PullRequestLabel.acc_id == CheckRun.acc_id,
                    PullRequestLabel.pull_request_node_id == CheckRun.pull_request_node_id,
                    func.lower(PullRequestLabel.name).in_(all_in_labels),
                )))
        else:
            only_prs = True
            # again, Postgres' planner sucks at checking this so we filter manually below
            # filters.append(CheckRun.pull_request_node_id.isnot(None))
    else:
        embedded_labels_query = False
    if not jira:
        query = select([CheckRun]).where(and_(*filters))
    else:
        query = await generate_jira_prs_query(
            filters, jira, None, mdb, cache, columns=CheckRun.__table__.columns, seed=CheckRun,
            on=(CheckRun.pull_request_node_id, CheckRun.acc_id))
    query = query \
        .with_statement_hint("IndexScan(c node_commit_repository_target)") \
        .with_statement_hint("IndexOnlyScan(c_1 github_node_commit_check_runs)") \
        .with_statement_hint("IndexOnlyScan(p github_node_push_redux)") \
        .with_statement_hint("IndexOnlyScan(prc node_pull_request_commit_commit_pr)") \
        .with_statement_hint("IndexScan(pr node_pullrequest_pkey)") \
        .with_statement_hint("IndexScan(sc ath_node_statuscontext_commit_created_at)") \
        .with_statement_hint("IndexScan(cr github_node_check_run_repository_started_at)") \
        .with_statement_hint("IndexOnlyScan(cs github_node_check_suite_created_at)") \
        .with_statement_hint("Rows(cr cs *400)") \
        .with_statement_hint("Rows(c_1 sc *1000)")

    df = await read_sql_query_with_join_collapse(
        query, mdb, CheckRun, soft_limit=maximum_processed_check_runs)

    # add check runs mapped to the mentioned PRs even if they are outside of the date range
    df, df_labels = await _append_pull_request_check_runs_outside(
        df, time_from, time_to, labels, embedded_labels_query, meta_ids, mdb)

    # the same check runs / suites may attach to different PRs, fix that
    df, *pr_labels = await _disambiguate_pull_requests(df, with_logical_repos, log, meta_ids, mdb)

    if only_prs:
        # we don't filter this in SQL (see above) so have to carve the result in code
        df = df.take(np.flatnonzero(df[CheckRun.pull_request_node_id.name].values))
        df.reset_index(inplace=True, drop=True)

    # deferred filter by labels so that we disambiguate PRs always the same way
    if labels:
        df = _filter_by_pr_labels(df, labels, embedded_labels_query, df_labels)

    # some status contexts represent the start and the finish events, join them together
    df_len = len(df)
    df = _merge_status_contexts(df)
    log.info("merged %d / %d", df_len - len(df), df_len)

    # "Re-run jobs" may produce duplicate check runs in the same check suite, split them
    # in separate artificial check suites by enumerating in chronological order
    df_len = len(df)
    df = _split_duplicate_check_runs(df)
    log.info("split %d / %d", len(df) - df_len, df_len)

    _postprocess_check_runs(df)

    if with_logical_repos:
        df = split_logical_repositories(
            df, *pr_labels, repositories, logical_settings, reindex=False, reset_index=False,
            repo_column=CheckRun.repository_full_name.name,
            id_column=CheckRun.pull_request_node_id.name,
            title_column=pull_request_title_column)
        df.reset_index(inplace=True, drop=True)

    return df, only_prs


@sentry_span
async def _disambiguate_pull_requests(df: pd.DataFrame,
                                      with_logical_repo_support: bool,
                                      log: logging.Logger,
                                      meta_ids: Tuple[int, ...],
                                      mdb: Database,
                                      ) -> Union[Tuple[pd.DataFrame],
                                                 Tuple[pd.DataFrame, pd.DataFrame]]:
    pr_node_ids = df[CheckRun.pull_request_node_id.name].values
    check_run_node_ids = df[CheckRun.check_run_node_id.name].values.astype(int, copy=False)
    unique_node_ids, node_id_counts = np.unique(check_run_node_ids, return_counts=True)
    ambiguous_unique_check_run_indexes = np.nonzero(node_id_counts > 1)[0]
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

    # we need all PR lifetimes to check explicit_node_id_indexes
    pr_cols = [NodePullRequest.id, NodePullRequest.author_id, NodePullRequest.merged,
               NodePullRequest.created_at, NodePullRequest.closed_at]
    if with_logical_repo_support:
        pr_cols.append(NodePullRequest.title)
    unique_pr_ids = np.unique(pr_node_ids[np.not_equal(pr_node_ids, None)])
    pr_lifetimes, pr_commit_counts, *pr_labels = await gather(
        read_sql_query(
            select(pr_cols)
            .where(and_(NodePullRequest.acc_id.in_(meta_ids),
                        NodePullRequest.id.in_any_values(unique_pr_ids))),
            mdb, pr_cols, index=NodePullRequest.id.name),
        read_sql_query(
            select([NodePullRequestCommit.pull_request_id,
                    func.count(NodePullRequestCommit.commit_id).label("count")])
            .where(and_(NodePullRequestCommit.acc_id.in_(meta_ids),
                        NodePullRequestCommit.pull_request_id.in_any_values(
                            unique_ambiguous_pr_node_ids)))
            .group_by(NodePullRequestCommit.pull_request_id),
            mdb, [NodePullRequestCommit.pull_request_id.name, "count"],
            index=NodePullRequestCommit.pull_request_id.name),
        *([fetch_labels_to_filter(unique_pr_ids, meta_ids, mdb)]
          if with_logical_repo_support else []),
    )
    del unique_ambiguous_pr_node_ids
    pr_lifetimes[NodePullRequest.closed_at.name].fillna(
        datetime.now(timezone.utc), inplace=True)
    df = df.join(
        pr_lifetimes[[
            NodePullRequest.created_at.name,
            NodePullRequest.closed_at.name,
            NodePullRequest.merged.name,
        ] + ([NodePullRequest.title.name] if with_logical_repo_support else [])],
        on=CheckRun.pull_request_node_id.name)
    df.rename(columns={
        NodePullRequest.created_at.name: pull_request_started_column,
        NodePullRequest.closed_at.name: pull_request_closed_column,
        NodePullRequest.merged.name: pull_request_merged_column,
        **({NodePullRequest.title.name: pull_request_title_column}
           if with_logical_repo_support else {}),
    }, inplace=True)
    df[pull_request_merged_column].fillna(False, inplace=True)
    if with_logical_repo_support:
        df[pull_request_title_column].fillna("", inplace=True)

    # do not let different check runs belonging to the same suite map to different PRs
    _calculate_check_suite_started(df)
    try:
        check_runs_outside_pr_lifetime_indexes = \
            np.flatnonzero(~df[check_suite_started_column].between(
                df[pull_request_started_column],
                df[pull_request_closed_column] + timedelta(hours=1),
            ).values)
    except TypeError:
        # "Cannot compare tz-naive and tz-aware datetime-like objects"
        # all the timestamps are NAT-s
        check_runs_outside_pr_lifetime_indexes = np.arange(len(df))
    # check run must launch while the PR remains open
    df[CheckRun.pull_request_node_id.name] = \
        df[CheckRun.pull_request_node_id.name].astype(object, copy=False)
    df[CheckRun.pull_request_node_id.name].values[check_runs_outside_pr_lifetime_indexes] = None
    old_df_len = len(df)
    """
    faster than
    df.drop_duplicates([CheckRun.check_run_node_id.name, CheckRun.pull_request_node_id.name],
                       inplace=True, ignore_index=True)
    """
    dupe_arr = np.zeros(len(df), dtype=[("cr", int), ("pr", int)])
    dupe_arr["cr"] = df[CheckRun.check_run_node_id.name].values
    nnz_mask = df[CheckRun.pull_request_node_id.name].notnull().values
    dupe_arr["pr"][nnz_mask] = \
        df[CheckRun.pull_request_node_id.name].values[nnz_mask].astype(int, copy=False)
    df = df.take(np.unique(dupe_arr, return_index=True)[1])
    df.reset_index(drop=True, inplace=True)
    log.info("rejecting check runs by PR lifetimes: %d / %d", len(df), old_df_len)

    if len(ambiguous_unique_check_run_indexes):
        # second lap
        check_run_node_ids = df[CheckRun.check_run_node_id.name].values
        pr_node_ids = df[CheckRun.pull_request_node_id.name].values.copy()
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
        ambiguous_df = pd.DataFrame({
            CheckRun.check_run_node_id.name: ambiguous_check_run_node_ids,
            CheckRun.check_suite_node_id.name:
                df[CheckRun.check_suite_node_id.name].values[ambiguous_indexes],
            CheckRun.pull_request_node_id.name: ambiguous_pr_node_ids,
            CheckRun.author_user_id.name:
                df[CheckRun.author_user_id.name].values[ambiguous_indexes],
        }).join(pr_lifetimes[[NodePullRequest.author_id.name, NodePullRequest.created_at.name]],
                on=CheckRun.pull_request_node_id.name)
        # we need to sort to stabilize idxmin() in step 2
        ambiguous_df.sort_values(NodePullRequest.created_at.name, inplace=True)
        # heuristic: the PR should be created by the commit author
        passed = np.nonzero((
            ambiguous_df[NodePullRequest.author_id.name] ==
            ambiguous_df[CheckRun.author_user_id.name]
        ).values)[0]
        log.info("disambiguation step 1 - authors: %d / %d", len(passed), len(ambiguous_df))
        passed_df = ambiguous_df.take(passed).join(
            pr_commit_counts, on=CheckRun.pull_request_node_id.name)
        del ambiguous_df
        # heuristic: the PR with the least number of commits wins
        order = np.argsort(passed_df["count"].values, kind="stable")
        passed_cr_node_ids = passed_df[CheckRun.check_run_node_id.name].values[order]
        _, first_encounters = np.unique(passed_cr_node_ids, return_index=True)
        passed = passed_df.index.values[order[first_encounters]].astype(int, copy=False)
        log.info("disambiguation step 2 - commit counts: %d / %d", len(passed), len(passed_df))
        del passed_df
        # we may discard some check runs completely here, set pull_request_node_id to None for them
        passed_mask = np.zeros_like(ambiguous_indexes, dtype=bool)
        passed_mask[passed] = True
        reset_indexes = ambiguous_indexes[~passed_mask]
        log.info("disambiguated null-s: %d / %d", len(reset_indexes), len(ambiguous_indexes))
        df.loc[reset_indexes, CheckRun.pull_request_node_id.name] = None
        # there can be check runs mapped to both a PR and None; remove None-s
        pr_node_ids[reset_indexes] = -1
        pr_node_ids[np.equal(pr_node_ids, None)] = -1
        pr_node_ids = pr_node_ids.astype(int, copy=False)
        joint = np.char.add(int_to_str(check_run_node_ids), int_to_str(pr_node_ids))
        order = np.argsort(joint)
        _, first_encounters = np.unique(check_run_node_ids[order], return_index=True)
        first_encounters = order[first_encounters]
        # first_encounters either map to a PR or to the only None for each check run
        log.info("final size: %d / %d", len(first_encounters), len(df))
        df = df.take(first_encounters)
        df.reset_index(inplace=True, drop=True)

    # cast pull_request_node_id to int
    pr_node_ids = df[CheckRun.pull_request_node_id.name].values
    pr_node_ids[np.equal(pr_node_ids, None)] = 0
    df[CheckRun.pull_request_node_id.name] = pr_node_ids.astype(int, copy=False)

    # TODO(vmarkovtsev): remove () after https://github.com/returntocorp/semgrep/issues/4461 is fixed  # noqa
    return (df, *pr_labels)


@sentry_span
def _erase_completed_at_as_needed(df: pd.DataFrame):
    if df.empty:
        return

    # exclude skipped checks from execution time calculation
    df[CheckRun.completed_at.name].values[df[CheckRun.conclusion.name].values == "NEUTRAL"] = None

    # exclude "too quick" checks from execution time calculation DEV-3155
    cr_types = np.empty(len(df), dtype=[("repo", int), ("name", np.uint64)])
    cr_types["repo"] = df[CheckRun.repository_node_id.name].values.byteswap()
    cr_names = df[CheckRun.name.name].values.astype("U")
    # take the hash to avoid very long names - they lead to bad performance
    cr_types["name"] = np.fromiter(
        (xxh3_64_intdigest(s) for s in cr_names.view(f"S{cr_names.dtype.itemsize}")),
        np.uint64, len(df))
    cr_types = cr_types.view("S16")  # speeds up np.unique()
    first_encounters, suite_groups = mark_check_suite_types(
        cr_types, df[CheckRun.check_suite_node_id.name].values)
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
        df[check_suite_completed_column].values - df[check_suite_started_column].values
    )[first_encounters[np.argsort(suite_groups)]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        median_cs_run_times = np.nanmedian(cs_run_times_shaped, axis=-1)
    excluded_cs_types = np.flatnonzero(median_cs_run_times <= np.timedelta64(8, "s"))
    excluded_cs = df[CheckRun.check_suite_node_id.name].values[
        first_encounters[np.in1d(suite_groups, excluded_cs_types)]
    ]
    excluded_mask = np.in1d(df[CheckRun.check_suite_node_id.name].values, excluded_cs)
    df[check_suite_completed_column].values[excluded_mask] = None


@sentry_span
def _postprocess_check_runs(df: pd.DataFrame) -> None:
    # there can be checks that finished before starting 🤦‍
    # pd.DataFrame.max(axis=1) does not work correctly because of the NaT-s
    started_ats = df[CheckRun.started_at.name].values
    df[CheckRun.completed_at.name] = np.maximum(df[CheckRun.completed_at.name].values, started_ats)
    df[CheckRun.completed_at.name] = df[CheckRun.completed_at.name].astype(started_ats.dtype)
    df[check_suite_completed_column] = df.groupby(
        CheckRun.check_suite_node_id.name, sort=False,
    )[CheckRun.completed_at.name].transform("max")

    _erase_completed_at_as_needed(df)

    # ensure that the timestamps are in sync after pruning PRs
    pr_ts_columns = [
        pull_request_started_column,
        pull_request_closed_column,
        pull_request_merged_column,
    ]
    df.loc[df[CheckRun.pull_request_node_id.name] == 0, pr_ts_columns] = None
    for column in pr_ts_columns:
        df.loc[df[column] == 0, column] = None
        df[column] = df[column].astype(started_ats.dtype, copy=False)

    for col in (CheckRun.check_run_node_id, CheckRun.check_suite_node_id,
                CheckRun.repository_node_id, CheckRun.commit_node_id):
        assert df[col.name].dtype == int, col.name


@sentry_span
async def _append_pull_request_check_runs_outside(df: pd.DataFrame,
                                                  time_from: datetime,
                                                  time_to: datetime,
                                                  labels: LabelFilter,
                                                  embedded_labels_query: bool,
                                                  meta_ids: Tuple[int, ...],
                                                  mdb: Database,
                                                  ) -> [pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    pr_node_ids = df[CheckRun.pull_request_node_id.name]
    prs_from_the_past = pr_node_ids[(
        df[CheckRun.pull_request_created_at.name] < (time_from + timedelta(days=1))
    ).values].unique()
    prs_to_the_future = pr_node_ids[(
        df[CheckRun.pull_request_closed_at.name].isnull() |
        (df[CheckRun.pull_request_closed_at.name] > (time_to - timedelta(days=1)))
    ).values].unique()
    query_before = select([CheckRunByPR]).where(and_(
        CheckRunByPR.acc_id.in_(meta_ids),
        CheckRunByPR.pull_request_node_id.in_(prs_from_the_past),
        CheckRunByPR.started_at.between(
            time_from - timedelta(days=90), time_from - timedelta(seconds=1),
        )))
    query_after = select([CheckRunByPR]).where(and_(
        CheckRunByPR.acc_id.in_(meta_ids),
        CheckRunByPR.pull_request_node_id.in_(prs_to_the_future),
        CheckRunByPR.started_at.between(
            time_to + timedelta(seconds=1), time_to + timedelta(days=90),
        )))
    pr_sql = union_all(query_before, query_after)
    tasks = [
        read_sql_query_with_join_collapse(pr_sql, mdb, CheckRunByPR),
    ]
    if labels and not embedded_labels_query:
        tasks.append(read_sql_query(
            select([PullRequestLabel.pull_request_node_id,
                    func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name)])
            .where(and_(PullRequestLabel.pull_request_node_id.in_(pr_node_ids.unique()),
                        PullRequestLabel.acc_id.in_(meta_ids))),
            mdb, [PullRequestLabel.pull_request_node_id, PullRequestLabel.name],
            index=PullRequestLabel.pull_request_node_id.name))
    extra_df, *df_labels = await gather(*tasks)
    for col in (CheckRun.committed_date_hack,
                CheckRun.pull_request_created_at,
                CheckRun.pull_request_closed_at):
        del df[col.name]
    if not extra_df.empty:
        df = df.append(extra_df, ignore_index=True)
    df[CheckRun.completed_at.name] = df[CheckRun.completed_at.name].astype(
        df[CheckRun.started_at.name].dtype)
    return df, df_labels


def _calculate_check_suite_started(df: pd.DataFrame) -> None:
    df[check_suite_started_column] = df.groupby(
        CheckRun.check_suite_node_id.name, sort=False,
    )[CheckRun.started_at.name].transform("min")


@sentry_span
def _filter_by_pr_labels(df: pd.DataFrame,
                         labels: LabelFilter,
                         embedded_labels_query: bool,
                         df_labels: Tuple[pd.DataFrame, ...],
                         ) -> pd.DataFrame:
    pr_node_ids = df[CheckRun.pull_request_node_id.name].values
    if not embedded_labels_query:
        df_labels = df_labels[0]
        prs_left = PullRequestMiner.find_left_by_labels(
            pd.Index(pr_node_ids),
            df_labels.index,
            df_labels[PullRequestLabel.name.name].values,
            labels,
        )
        indexes_left = np.nonzero(np.in1d(pr_node_ids, prs_left.values))[0]
        if len(indexes_left) < len(df):
            df = df.take(indexes_left)
            df.reset_index(drop=True, inplace=True)
    return df


@sentry_span
def _merge_status_contexts(df: pd.DataFrame) -> pd.DataFrame:
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
    statuses = df[CheckRun.status.name].values.astype("S", copy=False).copy()
    statuses[statuses == b"PENDING"] = b"A"
    starteds = df[CheckRun.started_at.name].values
    order = np.argsort(np.char.add(int_to_str(starteds.view(int)), statuses))
    df = df.take(order)
    statuses = statuses[order]
    starteds = starteds[order]
    del order
    df.reset_index(inplace=True, drop=True)
    # Important: we must sort in any case
    no_finish = np.flatnonzero(df[CheckRun.completed_at.name].isnull().values)
    if len(no_finish) == 0:
        # sweet dreams...
        return df
    no_finish_urls = df[CheckRun.url.name].values[no_finish].astype("S")
    empty_url_mask = no_finish_urls == b""
    empty_url_names = df[CheckRun.name.name].values[no_finish[empty_url_mask]].astype("U")
    no_finish_urls[empty_url_mask] = empty_url_names.view(f"S{empty_url_names.dtype.itemsize}")
    no_finish_parents = \
        df[CheckRun.check_suite_node_id.name].values[no_finish].astype(int, copy=False)
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
    matched_beg = no_finish_original_statuses[firsts] == b"A"  # replaced b"PENDING" earlier
    matched_end = counts > 1  # the status does not matter
    matched = matched_beg & matched_end
    matched_firsts = firsts[matched]
    matched_lasts = lasts[matched]

    # calculate the indexes of removed (merged) check runs
    drop_mask = np.zeros(len(no_finish_original), bool)
    lengths = matched_lasts - 1 - matched_firsts
    dropped = np.repeat(matched_lasts - 1 - lengths.cumsum(), lengths) + np.arange(lengths.sum())
    drop_mask[dropped] = True
    dropped = no_finish_original[drop_mask]

    matched_first_starteds = starteds[no_finish_original[matched_firsts]]
    matched_indexes = no_finish_original[matched_lasts]
    df[CheckRun.started_at.name].values[matched_indexes] = matched_first_starteds
    df[CheckRun.completed_at.name].values[matched_indexes] = starteds[matched_indexes]
    if len(dropped):
        df.drop(index=dropped, inplace=True)
        df.reset_index(inplace=True, drop=True)
    return df


@sentry_span
def _split_duplicate_check_runs(df: pd.DataFrame) -> pd.DataFrame:
    # DEV-2612 split older re-runs to artificial check suites
    # we require the df to be sorted by CheckRun.started_at
    if df.empty:
        return df
    dupe_index = df.groupby(
        [CheckRun.check_suite_node_id.name, CheckRun.name.name], sort=False,
    ).cumcount().values
    check_suite_node_ids = (dupe_index << (64 - 8)) | df[CheckRun.check_suite_node_id.name].values
    df[CheckRun.check_suite_node_id.name] = check_suite_node_ids
    check_run_conclusions = df[CheckRun.conclusion.name].values.astype("S")
    check_suite_conclusions = df[CheckRun.check_suite_conclusion.name].values
    successful = (
        (check_suite_conclusions == "SUCCESS") | (check_suite_conclusions == "NEUTRAL")
    )
    # override the successful conclusion of the check suite if at least one check run's conclusion
    # does not agree
    changed = False
    for c in ("TIMED_OUT", "CANCELLED", "FAILURE"):  # the order matters
        mask = successful & np.in1d(
            check_suite_node_ids,
            np.unique(check_suite_node_ids[check_run_conclusions == c.encode()]),
        )
        if mask.any():
            df.loc[mask, CheckRun.check_suite_conclusion.name] = c
            changed = True
    if changed:
        _calculate_check_suite_started(df)
        df.reset_index(inplace=True, drop=True)
    return df
