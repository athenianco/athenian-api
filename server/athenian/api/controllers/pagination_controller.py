import asyncio

from aiohttp import web
import numpy as np
from sqlalchemy import distinct, select

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.balancing import weight
from athenian.api.controllers.filter_controller import resolve_filter_prs_parameters
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.internal.account import get_user_account_status_from_request
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.precomputed_prs import DonePRFactsLoader
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.models.metadata.github import NodePullRequest, PullRequest
from athenian.api.models.web import (
    InvalidRequestError,
    PaginatePullRequestsRequest,
    PullRequestPaginationPlan,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.unordered_unique import unordered_unique
from athenian.precomputer.db.models import GitHubPullRequestDeployment


@weight(1)
async def paginate_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Compute the balanced pagination plan for `/filter/pull_requests`."""
    try:
        filt = PaginatePullRequestsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e))
    await get_user_account_status_from_request(request, filt.request.account)
    # we ignore events and stages because we cannot do anything with them
    (
        time_from,
        time_to,
        repos,
        _,
        _,
        participants,
        labels,
        jira,
        _,
        release_settings,
        logical_settings,
        prefixer,
        meta_ids,
    ) = await resolve_filter_prs_parameters(filt.request, request)
    ghprd = GitHubPullRequestDeployment
    deployed_prs_task = asyncio.create_task(
        # pessimistic: we ignore the environments
        read_sql_query(
            select(distinct(ghprd.pull_request_id)).where(
                ghprd.acc_id == filt.request.account,
                ghprd.repository_full_name.in_(repos),
                ghprd.finished_at.between(time_from, time_to),
            ),
            request.pdb,
            [ghprd.pull_request_id],
        ),
    )
    branches, default_branches = await BranchMiner.load_branches(
        repos, prefixer, filt.request.account, meta_ids, request.mdb, request.pdb, request.cache,
    )
    # we ignore the ambiguous PRs, thus producing a pessimistic prediction (that's OK)
    (done_ats, _), _ = await gather(
        DonePRFactsLoader.load_precomputed_done_timestamp_filters(
            time_from,
            time_to,
            repos,
            participants,
            labels,
            default_branches,
            filt.request.exclude_inactive,
            release_settings,
            prefixer,
            filt.request.account,
            request.pdb,
        ),
        deployed_prs_task,
    )
    done_node_ids = unordered_unique(
        np.fromiter((node_id for node_id, repo in done_ats), dtype=int, count=len(done_ats)),
    )
    del done_ats
    deployed_node_ids = deployed_prs_task.result()[ghprd.pull_request_id.name].values
    del deployed_prs_task
    if len(deployed_node_ids):
        done_node_ids = np.union1d(deployed_node_ids, done_node_ids)
    del deployed_node_ids
    add_pdb_hits(request.pdb, "paged_prs", len(done_node_ids))
    tasks = [
        PullRequestMiner.fetch_prs(
            time_from,
            time_to,
            repos,
            participants,
            labels,
            jira,
            filt.request.exclude_inactive,
            PullRequest.node_id.notin_any_values(done_node_ids),
            None,
            branches,
            None,
            filt.request.account,
            meta_ids,
            request.mdb,
            request.pdb,
            request.cache,
            columns=[PullRequest.node_id, PullRequest.updated_at],
        ),
        read_sql_query(
            select(NodePullRequest.node_id, NodePullRequest.updated_at).where(
                NodePullRequest.acc_id.in_(meta_ids),
                NodePullRequest.node_id.in_(done_node_ids),
            ),
            request.mdb,
            [NodePullRequest.node_id, NodePullRequest.updated_at],
        ),
    ]
    if jira:
        tasks.append(
            PullRequestMiner.filter_jira(
                done_node_ids,
                jira,
                meta_ids,
                request.mdb,
                request.cache,
                model=NodePullRequest,
                columns=[NodePullRequest.node_id],
            ),
        )
    (other_prs, *_), done_updated_at, *passed_jira = await gather(*tasks)
    if jira:
        done_updated_at = done_updated_at[NodePullRequest.updated_at.name].values[
            np.in1d(
                done_updated_at[NodePullRequest.node_id.name].values,
                passed_jira[0].index.values,
                assume_unique=True,
            )
        ]
    else:
        done_updated_at = done_updated_at[NodePullRequest.updated_at.name].values
    add_pdb_misses(request.pdb, "paged_prs", len(other_prs))
    if len(done_updated_at):
        updateds = np.concatenate(
            [other_prs[PullRequest.updated_at.name].values, done_updated_at],
            casting="unsafe",
        )
    elif len(other_prs) > 0:
        updateds = other_prs[PullRequest.updated_at.name].values
    else:
        updateds = np.array([time_from, filt.request.date_to], dtype="datetime64[ns]")
    updateds = np.sort(updateds.astype("datetime64[D]"))
    split = updateds[:: -filt.batch]
    split = np.concatenate([split, [updateds[0]]])  # append the other end
    split = np.unique(split)[::-1].tolist()  # there can be duplications
    if len(split) == 1:
        # all PRs happened on the same day
        split *= 2
    model = PullRequestPaginationPlan(updated=split)
    return model_response(model)
