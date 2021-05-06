from datetime import datetime, timezone
import logging
import pickle
from typing import Collection, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select, union_all

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.models.metadata.github import CheckRun, NodePullRequest, NodePullRequestCommit
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, commit_authors, jira, **_:  # noqa
    (
        time_from.timestamp(), time_to.timestamp(),
        ",".join(sorted(repositories)),
        ",".join(sorted(commit_authors)),
        jira,
    ),
)
async def mine_check_runs(time_from: datetime,
                          time_to: datetime,
                          repositories: Collection[str],
                          commit_authors: Collection[str],
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
    :param commit_authors: Check runs must link to the commits with the given author logins.
    :param jira: Check runs must link to PRs satisfying this JIRA filter.
    :return: Pandas DataFrame with columns mapped from CheckRun model.
    """
    filters = [
        CheckRun.acc_id.in_(meta_ids),
        CheckRun.started_at.between(time_from, time_to),
        CheckRun.repository_full_name.in_(repositories),
    ]
    if commit_authors:
        filters.append(CheckRun.author_login.in_(commit_authors))
    if not jira:
        query = select([CheckRun]).where(and_(*filters))
    else:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=CheckRun.__table__.columns, seed=CheckRun,
            on=(CheckRun.pull_request_node_id, CheckRun.acc_id))
    df = await read_sql_query(query, mdb, columns=CheckRun)

    # load check runs mapped to the mentioned PRs even if they are outside of the date range
    pr_node_ids = df[CheckRun.pull_request_node_id.key].unique()

    def filters():
        return [
            CheckRun.acc_id.in_(meta_ids),
            CheckRun.pull_request_node_id.in_any_values(pr_node_ids),
        ]

    query_before = select([CheckRun]).where(and_(*filters(), CheckRun.started_at < time_from))
    query_after = select([CheckRun]).where(and_(*filters(), CheckRun.started_at >= time_to))
    extra_df = await read_sql_query(union_all(query_before, query_after), mdb, columns=CheckRun)
    df = df.append(extra_df, ignore_index=True)
    del extra_df

    # another groupby() replacement for speed
    # we determine check runs belonging to the same commit with multiple pull requests
    node_ids = df[CheckRun.check_run_node_id.key].values.astype("S")
    unique_node_ids, node_id_counts = np.unique(node_ids, return_counts=True)
    ambiguous_node_id_indexes = np.nonzero((unique_node_ids != b"None") & (node_id_counts > 1))[0]
    if len(ambiguous_node_id_indexes):
        log = logging.getLogger(f"{metadata.__package__}.mine_check_runs")
        log.debug("Must disambiguate %d check runs", len(ambiguous_node_id_indexes))
        node_ids_order = np.argsort(node_ids)
        node_ids_group_counts = np.zeros(len(node_id_counts) + 1, dtype=int)
        np.cumsum(node_id_counts, out=node_ids_group_counts[1:])
        groups = np.array(np.split(node_ids_order, node_ids_group_counts[1:-1]))
        groups = groups[ambiguous_node_id_indexes]

        ambiguous_indexes = np.concatenate(groups)
        ambiguous_pr_node_ids = \
            df[CheckRun.pull_request_node_id.key].values.astype("U")[ambiguous_indexes]
        unique_ambiguous_pr_node_ids = np.unique(ambiguous_pr_node_ids)
        pr_cols = [NodePullRequest.id, NodePullRequest.author,
                   NodePullRequest.created_at, NodePullRequest.closed_at]
        pr_lifetimes, pr_commit_counts = await gather(
            read_sql_query(
                select(pr_cols)
                .where(and_(NodePullRequest.acc_id.in_(meta_ids),
                            NodePullRequest.id.in_any_values(unique_ambiguous_pr_node_ids))),
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
        pr_lifetimes[NodePullRequest.closed_at.key].fillna(
            datetime.now(timezone.utc), inplace=True)
        ambiguous_check_run_node_ids = node_ids[ambiguous_indexes].astype("U")
        ambiguous_df = pd.DataFrame({
            CheckRun.check_run_node_id.key: ambiguous_check_run_node_ids,
            CheckRun.pull_request_node_id.key: ambiguous_pr_node_ids,
            CheckRun.author_user.key: df[CheckRun.author_user.key][ambiguous_indexes],
            CheckRun.started_at.key: df[CheckRun.started_at.key][ambiguous_indexes],
        }).join(pr_lifetimes, on=CheckRun.pull_request_node_id.key)
        # heuristic 1: check run must launch while the PR is open
        passed = np.nonzero(ambiguous_df[CheckRun.started_at.key].between(
            ambiguous_df[NodePullRequest.created_at.key],
            ambiguous_df[NodePullRequest.closed_at.key],
        ).values)[0]
        log.info("Disambiguation step 1 - lifetimes: %d / %d", len(passed), len(ambiguous_df))
        ambiguous_df = ambiguous_df.take(passed)
        # heuristic 2: the PR should be created by the commit author
        passed = np.nonzero((
            ambiguous_df[NodePullRequest.author.key] == ambiguous_df[CheckRun.author_user.key]
        ).values)[0]
        log.info("Disambiguation step 2 - authors: %d / %d", len(passed), len(ambiguous_df))
        ambiguous_df = ambiguous_df.take(passed).join(
            pr_commit_counts, on=CheckRun.pull_request_node_id.key)
        # heuristic 3: the PR with the least number of commits wins
        passed = ambiguous_df.groupby(CheckRun.check_run_node_id.key)["count"].idxmin().values
        log.info("Disambiguation step 3 - commit counts: %d / %d", len(passed), len(ambiguous_df))
        # we may discard some check runs completely here, set pull_request_node_id to None for them
        passed_check_run_node_ids = \
            ambiguous_df[CheckRun.check_run_node_id.key].unique().astype("U")
        reset_mask = np.in1d(
            ambiguous_check_run_node_ids, passed_check_run_node_ids,
            assume_unique=True, invert=True)
        _, reset_indexes = np.unique(ambiguous_check_run_node_ids[reset_mask], return_index=True)
        log.info("Disambiguated null-s: %d / %d", len(reset_indexes), reset_mask.sum())
        reset_indexes = ambiguous_indexes[reset_mask][reset_indexes]
        removed_indexes = np.setdiff1d(ambiguous_indexes, passed, assume_unique=True)
        removed_indexes = np.setdiff1d(removed_indexes, reset_indexes, assume_unique=True)
        df.loc[reset_indexes, CheckRun.pull_request_node_id.key] = None
        df.drop(index=removed_indexes, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df
