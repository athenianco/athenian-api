from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select

from athenian.api.int_to_str import int_to_str
from athenian.api.internal.miners.github.check_run import (
    _calculate_check_suite_started,
    _finalize_check_runs,
    _merge_status_contexts,
    _postprocess_check_runs,
    _split_duplicate_check_runs,
    check_suite_started_column,
    pull_request_closed_column,
    pull_request_merged_column,
    pull_request_started_column,
    pull_request_title_column,
)
from athenian.api.models.metadata.github import CheckRun, NodePullRequest, NodePullRequestCommit
from athenian.api.to_object_arrays import as_bool, is_null


def _disambiguate_pull_requests(
    df: pd.DataFrame,
    log: logging.Logger,
    pr_lifetimes: pd.DataFrame,
    pr_commit_counts: pd.DataFrame,
) -> pd.DataFrame:
    with_logical_repo_support = False
    # cast pull_request_node_id to int
    pr_node_ids = df[CheckRun.pull_request_node_id.name].values
    pr_node_ids[is_null(pr_node_ids)] = 0
    df[CheckRun.pull_request_node_id.name] = pr_node_ids = pr_node_ids.astype(int, copy=False)

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
    (
        select(pr_cols).where(
            and_(NodePullRequest.acc_id.in_([1]), NodePullRequest.id.in_any_values(unique_pr_ids))
        ),
        select(
            [
                NodePullRequestCommit.pull_request_id,
                func.count(NodePullRequestCommit.commit_id).label("count"),
            ]
        )
        .where(
            and_(
                NodePullRequestCommit.acc_id.in_([1]),
                NodePullRequestCommit.pull_request_id.in_any_values(unique_ambiguous_pr_node_ids),
            )
        )
        .group_by(NodePullRequestCommit.pull_request_id),
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
        },
        inplace=True,
    )
    df = df.join(
        pr_lifetimes[
            [pull_request_merged_column]
            + ([pull_request_started_column, pull_request_closed_column] if fetch_pr_ts else [])
            + ([pull_request_title_column] if with_logical_repo_support else [])
        ],
        on=CheckRun.pull_request_node_id.name,
    )
    df[pull_request_closed_column].fillna(datetime.now(timezone.utc), inplace=True)
    df[pull_request_closed_column].values[df[pull_request_started_column].isnull().values] = None
    df[pull_request_merged_column] = as_bool(df[pull_request_merged_column].values)
    if with_logical_repo_support:
        df[pull_request_title_column].fillna("", inplace=True)

    # do not let different check runs belonging to the same suite map to different PRs
    _calculate_check_suite_started(df)
    try:
        check_runs_outside_pr_lifetime_indexes = np.flatnonzero(
            ~df[check_suite_started_column]
            .between(
                df[pull_request_started_column],
                df[pull_request_closed_column] + timedelta(hours=1),
            )
            .values
        )
    except TypeError:
        # "Cannot compare tz-naive and tz-aware datetime-like objects"
        # all the timestamps are NAT-s
        check_runs_outside_pr_lifetime_indexes = np.arange(len(df))
    # check run must launch while the PR remains open
    df[CheckRun.pull_request_node_id.name].values[check_runs_outside_pr_lifetime_indexes] = 0
    old_df_len = len(df)
    """
    faster than
    df.drop_duplicates([CheckRun.check_run_node_id.name, CheckRun.pull_request_node_id.name],
                       inplace=True, ignore_index=True)
    """
    dupe_arr = int_to_str(
        df[CheckRun.check_run_node_id.name].values, df[CheckRun.pull_request_node_id.name].values
    )
    _, not_dupes = np.unique(dupe_arr, return_index=True)
    check_run_node_ids = df[CheckRun.check_run_node_id.name].values[not_dupes]
    pr_node_ids = df[CheckRun.pull_request_node_id.name].values[not_dupes]
    check_suite_node_ids = df[CheckRun.check_suite_node_id.name].values[not_dupes]
    author_node_ids = df[CheckRun.author_user_id.name].values[not_dupes]
    pull_request_starteds = df[pull_request_started_column].values[not_dupes]
    log.info("rejecting check runs by PR lifetimes: %d / %d", len(not_dupes), old_df_len)

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
        ambiguous_df = pd.DataFrame(
            {
                CheckRun.check_run_node_id.name: ambiguous_check_run_node_ids,
                CheckRun.check_suite_node_id.name: check_suite_node_ids[ambiguous_indexes],
                CheckRun.pull_request_node_id.name: ambiguous_pr_node_ids,
                CheckRun.author_user_id.name: author_node_ids[ambiguous_indexes],
                pull_request_started_column: pull_request_starteds[ambiguous_indexes],
            }
        ).join(
            pr_lifetimes[[NodePullRequest.author_id.name]], on=CheckRun.pull_request_node_id.name
        )
        # we need to sort to stabilize idxmin() in step 2
        ambiguous_df.sort_values(pull_request_started_column, inplace=True)
        # heuristic: the PR should be created by the commit author
        passed = np.flatnonzero(
            (
                ambiguous_df[NodePullRequest.author_id.name]
                == ambiguous_df[CheckRun.author_user_id.name]
            ).values
        )
        log.info("disambiguation step 1 - authors: %d / %d", len(passed), len(ambiguous_df))
        ambiguous_df.disable_consolidate()
        passed_df = ambiguous_df.take(passed).join(
            pr_commit_counts, on=CheckRun.pull_request_node_id.name
        )
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
        df[CheckRun.pull_request_node_id.name].values[not_dupes[reset_indexes]] = 0
        # there can be check runs mapped to both a PR and None; remove None-s
        pr_node_ids[reset_indexes] = -1
        pr_node_ids[pr_node_ids == 0] = -1
        joint = int_to_str(check_run_node_ids, pr_node_ids)
        order = np.argsort(joint)
        _, first_encounters = np.unique(check_run_node_ids[order], return_index=True)
        first_encounters = order[first_encounters]
        # first_encounters either map to a PR or to the only None for each check run
        log.info("final size: %d / %d", len(first_encounters), len(df))
        df.disable_consolidate()
        df = df.take(not_dupes[first_encounters])
        df.reset_index(inplace=True, drop=True)
        df.disable_consolidate()

    return df


def test_mine_check_runs_benchmark(benchmark, no_deprecation_warnings):
    with open("/tmp/dis1.pickle", "rb") as fin:
        df = pickle.load(fin)
    with open("/tmp/dis2.pickle", "rb") as fin:
        pr_lifetimes, pr_commit_counts, pr_labels = pickle.load(fin)
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
    split = _split_duplicate_check_runs(df)
    log.info("split %d / %d", split, len(df))

    _postprocess_check_runs(df)


def test_mine_check_runs_wrap(benchmark, no_deprecation_warnings):
    df = pd.read_csv(
        Path(__file__).parent.parent.parent / "features" / "github" / "check_runs.csv.gz",
        index_col=0,
    )
    for col in (
        CheckRun.started_at,
        CheckRun.completed_at,
        CheckRun.pull_request_created_at,
        CheckRun.pull_request_closed_at,
        CheckRun.committed_date,
        check_suite_started_column,
    ):
        col_name = col.name if not isinstance(col, str) else col
        df[col_name] = df[col_name].astype(np.datetime64)
    benchmark(_finalize_check_runs, df, logging.getLogger("pytest.alternative_facts"))
