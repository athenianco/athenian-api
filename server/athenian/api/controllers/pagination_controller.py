import asyncio
from datetime import timedelta

from aiohttp import web
import numpy as np

from athenian.api import ResponseError
from athenian.api.controllers.filter_controller import resolve_filter_prs_parameters
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_precomputed_done_timestamp_filters
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.web import InvalidRequestError, PaginatePullRequestsRequest, \
    PullRequestPaginationPlan
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response


async def paginate_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Compute the balanced pagination plan for `/filter/pull_requests`."""
    try:
        filt = PaginatePullRequestsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    # we ignore events and stages because we cannot do anything with them
    repos, _, _, participants, labels, jira, settings = \
        await resolve_filter_prs_parameters(filt.request, request)
    branches, default_branches = await extract_branches(repos, request.mdb, request.cache)
    done_ats = await load_precomputed_done_timestamp_filters(
        filt.request.date_from, filt.request.date_to, repos, participants, labels,
        default_branches, filt.request.exclude_inactive, settings, request.pdb)
    tasks = [
        PullRequestMiner.fetch_prs(
            filt.request.date_from, filt.request.date_to, repos, participants, labels, jira, 0,
            filt.request.exclude_inactive, PullRequest.node_id.notin_(done_ats), request.mdb,
            columns=[PullRequest.node_id, PullRequest.updated_at]),
    ]
    if filt.request.jira:
        tasks.append(PullRequestMiner.filter_jira(
            done_ats, jira, request.mdb, columns=[PullRequest.node_id]))
        other_prs, passed_jira = await asyncio.gather(*tasks, return_exceptions=True)
        for r in other_prs, passed_jira:
            if isinstance(r, Exception):
                raise r from None
        done_ats = {k: done_ats[k] for k in passed_jira.index.values}
    else:
        other_prs = await tasks[0]
    if done_ats:
        updateds = np.concatenate([other_prs[PullRequest.updated_at.key].values,
                                   np.asarray(list(done_ats.values()))]).astype("datetime64[ns]")
    elif len(other_prs) > 0:
        updateds = other_prs[PullRequest.updated_at.key].values
    else:
        updateds = np.array([filt.request.date_from, filt.request.date_to - timedelta(days=1)],
                            dtype="datetime64[ns]")
    updateds = np.sort(updateds.astype("datetime64[D]"))
    split = updateds[::-filt.batch]
    split[0] += np.timedelta64(1, "D")
    if updateds[0] != split[-1]:
        split = np.concatenate([split, [updateds[0]]])
    split = np.unique(split)[::-1]  # there can be duplications
    model = PullRequestPaginationPlan(updated=split.tolist())
    return model_response(model)
