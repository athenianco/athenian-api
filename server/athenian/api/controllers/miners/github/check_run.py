from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
from typing import Collection, Optional, Tuple, Type, Union

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, exists, func, select, union_all
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, short_term_exptime
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.db import DatabaseLike, ParallelDatabase
from athenian.api.models.metadata.github import CheckRun, CheckRunByPR, NodePullRequest, \
    NodePullRequestCommit, \
    NodeRepository, PullRequestLabel
from athenian.api.tracing import sentry_span


check_suite_started_column = "check_suite_started"
pull_request_started_column = "pull_request_" + NodePullRequest.created_at.key
pull_request_closed_column = "pull_request_" + NodePullRequest.closed_at.key
pull_request_merged_column = "pull_request_" + NodePullRequest.merged.key


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, pushers, labels, jira, **_:  # noqa
    (
        time_from.timestamp(), time_to.timestamp(),
        ",".join(sorted(repositories)),
        ",".join(sorted(pushers)),
        labels,
        jira,
    ),
)
async def mine_check_runs(time_from: datetime,
                          time_to: datetime,
                          repositories: Collection[str],
                          pushers: Collection[str],
                          labels: LabelFilter,
                          jira: JIRAFilter,
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
    :param repositories: Look for check runs in these repository names.
    :param pushers: Check runs must link to the commits with the given pusher logins.
    :param labels: Check runs must link to PRs marked with these labels.
    :param jira: Check runs must link to PRs satisfying this JIRA filter.
    :return: Pandas DataFrame with columns mapped from CheckRun model.
    """
    log = logging.getLogger(f"{metadata.__package__}.mine_check_runs")
    assert time_from.tzinfo is not None
    assert time_to.tzinfo is not None
    repo_nodes = [r[0] for r in await mdb.fetch_all(
        select([NodeRepository.id])
        .where(and_(NodeRepository.acc_id.in_(meta_ids),
                    NodeRepository.name_with_owner.in_(repositories))))]
    filters = [
        CheckRun.acc_id.in_(meta_ids),
        CheckRun.started_at.between(time_from, time_to),
        CheckRun.committed_date_hack.between(
            time_from - timedelta(days=14), time_to + timedelta(days=1)),
        CheckRun.repository_node_id.in_(repo_nodes),
    ]
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
            filters.append(CheckRun.pull_request_node_id.isnot(None))
    else:
        embedded_labels_query = False
    if not jira:
        query = select([CheckRun]).where(and_(*filters))
        set_join_collapse_limit = mdb.url.dialect == "postgresql"
    else:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=CheckRun.__table__.columns, seed=CheckRun,
            on=(CheckRun.pull_request_node_id, CheckRun.acc_id))
        set_join_collapse_limit = False
    df = await _read_sql_query_with_join_collapse(query, CheckRun, set_join_collapse_limit, mdb)
    # add check runs mapped to the mentioned PRs even if they are outside of the date range
    df, df_labels = await _append_pull_request_check_runs_outside(
        df, time_from, time_to, labels, embedded_labels_query, meta_ids, mdb)

    pr_node_ids = df[CheckRun.pull_request_node_id.key].values.astype("U")
    check_run_node_ids = df[CheckRun.check_run_node_id.key].values.astype("S")
    unique_node_ids, node_id_counts = np.unique(check_run_node_ids, return_counts=True)
    assert (unique_node_ids != b"None").all()
    ambiguous_unique_check_run_indexes = np.nonzero(node_id_counts > 1)[0]
    if len(ambiguous_unique_check_run_indexes):
        log.debug("Must disambiguate %d check runs", len(ambiguous_unique_check_run_indexes))
        # another groupby() replacement for speed
        # we determine check runs belonging to the same commit with multiple pull requests
        node_ids_order = np.argsort(check_run_node_ids)
        node_ids_group_counts = np.cumsum(node_id_counts)
        groups = np.array(np.split(node_ids_order, node_ids_group_counts[:-1]), dtype=object)
        groups = groups[ambiguous_unique_check_run_indexes]
        unique_ambiguous_pr_node_ids = np.unique(pr_node_ids[np.concatenate(groups)])
    else:
        unique_ambiguous_pr_node_ids = []

    # we need all PR lifetimes to check explicit_node_id_indexes
    pr_cols = [NodePullRequest.id, NodePullRequest.author, NodePullRequest.merged,
               NodePullRequest.created_at, NodePullRequest.closed_at]
    pr_lifetimes, pr_commit_counts = await gather(
        read_sql_query(
            select(pr_cols)
            .where(and_(NodePullRequest.acc_id.in_(meta_ids),
                        NodePullRequest.id.in_any_values(np.unique(pr_node_ids)))),
            mdb, pr_cols, index=NodePullRequest.id.key),
        read_sql_query(
            select([NodePullRequestCommit.pull_request,
                    func.count(NodePullRequestCommit.commit).label("count")])
            .where(and_(NodePullRequestCommit.acc_id.in_(meta_ids),
                        NodePullRequestCommit.pull_request.in_any_values(
                            unique_ambiguous_pr_node_ids)))
            .group_by(NodePullRequestCommit.pull_request),
            mdb, [NodePullRequestCommit.pull_request.key, "count"],
            index=NodePullRequestCommit.pull_request.key),
    )
    del unique_ambiguous_pr_node_ids
    pr_lifetimes[NodePullRequest.closed_at.key].fillna(
        datetime.now(timezone.utc), inplace=True)
    df = df.join(
        pr_lifetimes[[
            NodePullRequest.created_at.key,
            NodePullRequest.closed_at.key,
            NodePullRequest.merged.key,
        ]],
        on=CheckRun.pull_request_node_id.key)
    df.rename(columns={
        NodePullRequest.created_at.key: pull_request_started_column,
        NodePullRequest.closed_at.key: pull_request_closed_column,
        NodePullRequest.merged.key: pull_request_merged_column,
    }, inplace=True)
    df[pull_request_merged_column].fillna(False, inplace=True)

    # do not let different check runs belonging to the same suite map to different PRs
    _calculate_check_suite_started(df)
    try:
        check_runs_outside_pr_lifetime_indexes = \
            np.nonzero(~df[check_suite_started_column].between(
                df[pull_request_started_column],
                df[pull_request_closed_column] + timedelta(hours=1),
            ).values)[0]
    except TypeError:
        # "Cannot compare tz-naive and tz-aware datetime-like objects"
        # all the timestamps are NAT-s
        check_runs_outside_pr_lifetime_indexes = np.arange(len(df))
    # check run must launch while the PR remains open
    df.loc[check_runs_outside_pr_lifetime_indexes, CheckRun.pull_request_node_id.key] = None
    old_df_len = len(df)
    df.drop_duplicates([CheckRun.check_run_node_id.key, CheckRun.pull_request_node_id.key],
                       inplace=True, ignore_index=True)
    log.info("Rejecting check runs by PR lifetimes: %d / %d", len(df), old_df_len)

    if len(ambiguous_unique_check_run_indexes):
        # second lap
        check_run_node_ids = df[CheckRun.check_run_node_id.key].values.astype("S")
        pr_node_ids = df[CheckRun.pull_request_node_id.key].values.astype("S")
        unique_node_ids, node_id_counts = np.unique(check_run_node_ids, return_counts=True)
        ambiguous_unique_check_run_indexes = np.nonzero(node_id_counts > 1)[0]
        if len(ambiguous_unique_check_run_indexes):
            node_ids_order = np.argsort(check_run_node_ids)
            node_ids_group_counts = np.cumsum(node_id_counts)
            groups = np.array(np.split(node_ids_order, node_ids_group_counts[:-1]), dtype=object)
            groups = groups[ambiguous_unique_check_run_indexes]
            ambiguous_indexes = np.concatenate(groups)
            log.info("Must disambiguate %d check runs", len(ambiguous_indexes))
            ambiguous_pr_node_ids = pr_node_ids[ambiguous_indexes].astype("U")
    if len(ambiguous_unique_check_run_indexes):
        ambiguous_check_run_node_ids = check_run_node_ids[ambiguous_indexes].astype("U")
        ambiguous_df = pd.DataFrame({
            CheckRun.check_run_node_id.key: ambiguous_check_run_node_ids,
            CheckRun.check_suite_node_id.key:
                df[CheckRun.check_suite_node_id.key].values[ambiguous_indexes],
            CheckRun.pull_request_node_id.key: ambiguous_pr_node_ids,
            CheckRun.author_user.key: df[CheckRun.author_user.key].values[ambiguous_indexes],
        }).join(pr_lifetimes[[NodePullRequest.author.key, NodePullRequest.created_at.key]],
                on=CheckRun.pull_request_node_id.key)
        # we need to sort to stabilize idxmin() in step 2
        ambiguous_df.sort_values(NodePullRequest.created_at.key, inplace=True)
        # heuristic: the PR should be created by the commit author
        passed = np.nonzero((
            ambiguous_df[NodePullRequest.author.key] == ambiguous_df[CheckRun.author_user.key]
        ).values)[0]
        log.info("Disambiguation step 1 - authors: %d / %d", len(passed), len(ambiguous_df))
        passed_df = ambiguous_df.take(passed).join(
            pr_commit_counts, on=CheckRun.pull_request_node_id.key)
        del ambiguous_df
        # heuristic: the PR with the least number of commits wins
        passed = passed_df.groupby(CheckRun.check_run_node_id.key)["count"] \
            .idxmin().values.astype(int, copy=False)
        log.info("Disambiguation step 2 - commit counts: %d / %d", len(passed), len(passed_df))
        del passed_df
        # we may discard some check runs completely here, set pull_request_node_id to None for them
        passed_mask = np.zeros_like(ambiguous_indexes, dtype=bool)
        passed_mask[passed] = True
        reset_indexes = ambiguous_indexes[~passed_mask]
        log.info("Disambiguated null-s: %d / %d", len(reset_indexes), len(ambiguous_indexes))
        df.loc[reset_indexes, CheckRun.pull_request_node_id.key] = None
        # there can be check runs mapped to both a PR and None; remove None-s
        pr_node_ids[reset_indexes] = b"None"
        pr_node_ids[pr_node_ids == b"None"] = b"{"  # > a-zA-Z0-9
        joint = np.char.add(check_run_node_ids, pr_node_ids)
        order = np.argsort(joint)
        _, first_encounters = np.unique(check_run_node_ids[order], return_index=True)
        first_encounters = order[first_encounters]
        # first_encounters either map to a PR or to the only None for each check run
        log.info("Final size: %d / %d", len(first_encounters), len(df))
        df = df.take(first_encounters)
        df.reset_index(inplace=True, drop=True)

    # deferred filter by labels so that we disambiguate PRs always the same way
    if labels:
        df = _filter_by_pr_labels(df, labels, embedded_labels_query, df_labels)

    # some status contexts represent the start and the finish events, join them together
    _merge_status_contexts(df)

    # "Re-run jobs" may produce duplicate check runs in the same check suite, split them
    # in separate artificial check suites by enumerating in chronological order
    _split_duplicate_check_runs(df)

    # exclude skipped checks from execution time calculation
    df.loc[df[CheckRun.conclusion.key] == "NEUTRAL", CheckRun.completed_at.key] = None

    # ensure that the timestamps are in sync after pruning PRs
    pr_ts_columns = [
        pull_request_started_column,
        pull_request_closed_column,
        pull_request_merged_column,
    ]
    df.loc[df[CheckRun.pull_request_node_id.key].isnull(), pr_ts_columns] = None
    for column in pr_ts_columns:
        df.loc[df[column] == 0, column] = None

    # there can be checks that finished before starting 🤦‍
    # pd.DataFrame.max(axis=1) does not work correctly because of the NaT-s
    started_ats = df[CheckRun.started_at.key].values
    df[CheckRun.completed_at.key] = np.maximum(
        df[CheckRun.completed_at.key].fillna(pd.NaT).values, started_ats)
    df[CheckRun.completed_at.key] = df[CheckRun.completed_at.key].astype(started_ats.dtype)

    return df


@sentry_span
async def _append_pull_request_check_runs_outside(df: pd.DataFrame,
                                                  time_from: datetime,
                                                  time_to: datetime,
                                                  labels: LabelFilter,
                                                  embedded_labels_query: bool,
                                                  meta_ids: Tuple[int, ...],
                                                  mdb: ParallelDatabase,
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
    tasks = [
        _read_sql_query_with_join_collapse(
            union_all(query_before, query_after), CheckRunByPR, True, mdb),
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
    del df[CheckRun.committed_date_hack.name]
    del df[CheckRun.pull_request_created_at.name]
    del df[CheckRun.pull_request_closed_at.name]
    return df.append(extra_df, ignore_index=True), df_labels


async def _read_sql_query_with_join_collapse(query: ClauseElement,
                                             columns: Union[Type[CheckRun], Type[CheckRunByPR]],
                                             set_join_collapse_limit: bool,
                                             mdb: ParallelDatabase,
                                             ) -> pd.DataFrame:
    set_join_collapse_limit &= mdb.url.dialect == "postgresql"
    async with mdb.connection() as mdb_conn:
        if set_join_collapse_limit:
            transaction = mdb_conn.transaction()
            await transaction.start()
            await mdb_conn.execute("set join_collapse_limit=1")
        try:
            return await read_sql_query(query, mdb_conn, columns=columns)
        finally:
            if set_join_collapse_limit:
                await transaction.rollback()


def _calculate_check_suite_started(df: pd.DataFrame) -> None:
    df[check_suite_started_column] = df.groupby(
        CheckRun.check_suite_node_id.name, sort=False,
    )[CheckRun.started_at.name].transform("min")


def _filter_by_pr_labels(df: pd.DataFrame,
                         labels: LabelFilter,
                         embedded_labels_query: bool,
                         df_labels: Tuple[pd.DataFrame, ...],
                         ) -> pd.DataFrame:
    pr_node_ids = df[CheckRun.pull_request_node_id.key].values.astype("S")
    if not embedded_labels_query:
        df_labels = df_labels[0]
        prs_left = PullRequestMiner.find_left_by_labels(
            df_labels.index, df_labels[PullRequestLabel.name.key].values, labels)
        indexes_left = np.nonzero(np.in1d(pr_node_ids, prs_left.values.astype("S")))[0]
        if len(indexes_left) < len(df):
            df = df.take(indexes_left)
            df.reset_index(drop=True, inplace=True)
    return df


def _merge_status_contexts(df: pd.DataFrame) -> None:
    if df.empty:
        return
    no_finish = np.flatnonzero(df[CheckRun.completed_at.name].isnull().values)
    if len(no_finish) == 0:
        return
    no_finish_urls = df[CheckRun.url.name].values[no_finish].astype("S")
    no_finish_parents = df[CheckRun.check_suite_node_id.name].values[no_finish].astype("S")  # FIXME: str -> int  # noqa
    no_finish_seeds = np.char.add(no_finish_parents, no_finish_urls)
    _, first_encounters, indexes, counts = np.unique(
        no_finish_seeds, return_index=True, return_inverse=True, return_counts=True)
    masks = np.zeros((len(first_encounters), len(no_finish_seeds)), bool)
    arr_y = np.repeat(np.arange(len(first_encounters)), counts)
    arr_x = np.argsort(indexes)
    masks[arr_y, arr_x] = True
    timestamps = np.broadcast_to(df[CheckRun.started_at.name].values[no_finish][None, :],
                                 masks.shape)
    captured = counts > 1
    mins = np.min(timestamps, where=masks, initial=np.nanmax(timestamps[0]), axis=1)[captured]
    maxs = np.max(timestamps, where=masks, initial=np.nanmin(timestamps[0]), axis=1)
    argmaxs = np.argmax(np.equal(timestamps, maxs[:, None], where=masks), axis=1)[captured]
    maxs = maxs[captured]
    statuses = df[CheckRun.status.name].values[no_finish][argmaxs]
    # we have to cast dtype here to prevent changing dtype in df to object
    mins = pd.Series(mins).astype(df[CheckRun.started_at.name].dtype)
    maxs = pd.Series(maxs).astype(df[CheckRun.started_at.name].dtype)
    first_encounters = first_encounters[captured]
    first_no_finish = no_finish[first_encounters]
    mins.index = maxs.index = first_no_finish
    df.loc[first_no_finish, CheckRun.started_at.name] = mins
    df.loc[first_no_finish, CheckRun.completed_at.name] = maxs
    df.loc[first_no_finish, CheckRun.status.name] = statuses
    secondary = no_finish[np.setdiff1d(
        np.flatnonzero(np.in1d(indexes, np.flatnonzero(captured))),
        first_encounters,
        assume_unique=True)]
    df.drop(index=secondary, inplace=True)
    df.reset_index(inplace=True, drop=True)


def _split_duplicate_check_runs(df: pd.DataFrame) -> None:
    # DEV-2612 split older re-runs to artificial check suites
    if df.empty:
        return
    df.sort_values(CheckRun.started_at.name, ignore_index=True, inplace=True)
    dupe_index = df.groupby(
        [CheckRun.check_suite_node_id.name, CheckRun.name.name], sort=False,
    ).cumcount().values
    # astype("S") mistakes x10 in the length for unknown reason
    dupe_index = dupe_index.astype(f"S{len(str(dupe_index.max()))}")
    check_suite_node_ids = df[CheckRun.check_suite_node_id.key].values.astype("S")
    check_suite_node_ids = np.char.add(check_suite_node_ids, dupe_index)
    df[CheckRun.check_suite_node_id.key] = check_suite_node_ids
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
