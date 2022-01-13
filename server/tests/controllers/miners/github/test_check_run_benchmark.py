from datetime import datetime, timedelta, timezone
import logging
import pickle

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select

from athenian.api.async_utils import wrap_sql_query
from athenian.api.controllers.miners.github.check_run import _calculate_check_suite_started, \
    _merge_status_contexts, _postprocess_check_runs, _split_duplicate_check_runs, \
    check_suite_started_column, \
    pull_request_closed_column, \
    pull_request_merged_column, pull_request_started_column
from athenian.api.int_to_str import int_to_str
from athenian.api.models.metadata.github import CheckRun, NodePullRequest, NodePullRequestCommit


def _disambiguate_pull_requests(df: pd.DataFrame,
                                log: logging.Logger,
                                pr_lifetimes: pd.DataFrame,
                                pr_commit_counts: pd.DataFrame,
                                ) -> pd.DataFrame:
    pr_node_ids = df[CheckRun.pull_request_node_id.name].values
    check_run_node_ids = df[CheckRun.check_run_node_id.name].values.astype(int, copy=False)
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

    # we need all PR lifetimes to check explicit_node_id_indexes
    pr_cols = [NodePullRequest.id, NodePullRequest.author_id, NodePullRequest.merged,
               NodePullRequest.created_at, NodePullRequest.closed_at]
    (
        select(pr_cols)
        .where(and_(NodePullRequest.acc_id.in_([1]),
                    NodePullRequest.id.in_any_values(
                        np.unique(pr_node_ids[np.not_equal(pr_node_ids, None)]))))
    )
    (
        select([NodePullRequestCommit.pull_request_id,
                func.count(NodePullRequestCommit.commit_id).label("count")])
        .where(and_(NodePullRequestCommit.acc_id.in_([1]),
                    NodePullRequestCommit.pull_request_id.in_any_values(
                        unique_ambiguous_pr_node_ids)))
        .group_by(NodePullRequestCommit.pull_request_id)
    )
    del unique_ambiguous_pr_node_ids
    pr_lifetimes[NodePullRequest.closed_at.name].fillna(
        datetime.now(timezone.utc), inplace=True)
    df = df.join(
        pr_lifetimes[[
            NodePullRequest.created_at.name,
            NodePullRequest.closed_at.name,
            NodePullRequest.merged.name,
        ]],
        on=CheckRun.pull_request_node_id.name)
    df.rename(columns={
        NodePullRequest.created_at.name: pull_request_started_column,
        NodePullRequest.closed_at.name: pull_request_closed_column,
        NodePullRequest.merged.name: pull_request_merged_column,
    }, inplace=True)
    df[pull_request_merged_column].fillna(False, inplace=True)

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
        ambiguous_unique_check_run_indexes = np.nonzero(node_id_counts > 1)[0]
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
        """
        passed = passed_df.groupby(CheckRun.check_run_node_id.name)["count"] \
            .idxmin().values.astype(int, copy=False)
        """
        order = np.argsort(passed_df["count"].values, kind="stable")
        passed_cr_node_ids = passed_df[CheckRun.check_run_node_id.name].values[order]
        _, first_encounters = np.unique(passed_cr_node_ids, return_index=True)
        passed = passed_df.index.values[order[first_encounters]]
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

    return df


def test_mine_check_runs_benchmark(benchmark, no_deprecation_warnings):
    with open("/tmp/df1.pickle", "rb") as fin:
        df = pickle.load(fin)
    with open("/tmp/pr_lifetimes.pickle", "rb") as fin:
        pr_lifetimes = pickle.load(fin)
    with open("/tmp/pr_commit_counts.pickle", "rb") as fin:
        pr_commit_counts = pickle.load(fin)
    log = logging.getLogger("benchmark")
    benchmark(_benchmark_postprocess, df, log, pr_lifetimes, pr_commit_counts)


def _benchmark_postprocess(df, log, pr_lifetimes, pr_commit_counts):
    # the same check runs / suites may attach to different PRs, fix that
    df = _disambiguate_pull_requests(df, log, pr_lifetimes, pr_commit_counts)

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


def test_mine_check_runs_wrap(benchmark, no_deprecation_warnings):
    with open("/tmp/cr_raw.pickle", "rb") as fin:
        data, columns, index = pickle.load(fin)
    benchmark(wrap_sql_query, data, columns, index)
