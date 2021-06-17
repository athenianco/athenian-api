from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
from typing import Collection, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, exists, func, select, union_all

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import CheckRun, NodePullRequest, NodePullRequestCommit, \
    NodeRepository, PullRequestLabel
from athenian.api.tracing import sentry_span


check_suite_started_column = "check_suite_started"
pull_request_started_column = "pull_request_" + NodePullRequest.created_at.key
pull_request_closed_column = "pull_request_" + NodePullRequest.closed_at.key
pull_request_merged_column = "pull_request_" + NodePullRequest.merged.key


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
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
    repo_nodes = [r[0] for r in await mdb.fetch_all(
        select([NodeRepository.id])
        .where(and_(NodeRepository.acc_id.in_(meta_ids),
                    NodeRepository.name_with_owner.in_(repositories))))]
    filters = [
        CheckRun.acc_id.in_(meta_ids),
        CheckRun.started_at.between(time_from, time_to),
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
    if not jira:
        query = select([CheckRun]).where(and_(*filters))
    else:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=CheckRun.__table__.columns, seed=CheckRun,
            on=(CheckRun.pull_request_node_id, CheckRun.acc_id))
    df = await read_sql_query(query, mdb, columns=CheckRun)

    pr_node_ids = df[CheckRun.pull_request_node_id.key].unique()

    # load check runs mapped to the mentioned PRs even if they are outside of the date range
    def filters():
        return [
            CheckRun.acc_id.in_(meta_ids),
            CheckRun.pull_request_node_id.in_(pr_node_ids),
        ]

    query_before = select([CheckRun]).where(and_(*filters(), CheckRun.started_at < time_from))
    query_after = select([CheckRun]).where(and_(*filters(), CheckRun.started_at >= time_to))
    tasks = [
        read_sql_query(union_all(query_before, query_after), mdb, columns=CheckRun),
    ]
    if labels and not embedded_labels_query:
        tasks.append(read_sql_query(
            select([PullRequestLabel.pull_request_node_id,
                    func.lower(PullRequestLabel.name).label(PullRequestLabel.name.key)])
            .where(and_(PullRequestLabel.pull_request_node_id.in_(pr_node_ids),
                        PullRequestLabel.acc_id.in_(meta_ids))),
            mdb, [PullRequestLabel.pull_request_node_id, PullRequestLabel.name],
            index=PullRequestLabel.pull_request_node_id.key))
    extra_df, *df_labels = await gather(*tasks)
    df = df.append(extra_df, ignore_index=True)
    del extra_df

    pr_node_ids = df[CheckRun.pull_request_node_id.key].values.astype("U")
    if labels and not embedded_labels_query:
        df_labels = df_labels[0]
        prs_left = PullRequestMiner.find_left_by_labels(
            df_labels.index, df_labels[PullRequestLabel.name.key].values, labels)
        indexes_left = np.nonzero(np.in1d(pr_node_ids, prs_left.values.astype("U")))[0]
        if len(indexes_left) < len(df):
            pr_node_ids = pr_node_ids[indexes_left]
            df = df.take(indexes_left)
            df.reset_index(drop=True, inplace=True)

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
    df[check_suite_started_column] = df.groupby(
        CheckRun.check_suite_node_id.key, sort=False,
    )[CheckRun.started_at.key].transform("min")
    check_runs_outside_pr_lifetime_indexes = np.nonzero(~df[check_suite_started_column].between(
        df[pull_request_started_column],
        df[pull_request_closed_column] + timedelta(hours=1),
    ).values)[0]
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
