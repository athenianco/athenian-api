from aiohttp import web
import numpy as np

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.filter_controller import resolve_filter_prs_parameters
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_precomputed_done_timestamp_filters
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.web import InvalidRequestError, PaginatePullRequestsRequest, \
    PullRequestPaginationPlan
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@weight(1)
async def paginate_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Compute the balanced pagination plan for `/filter/pull_requests`."""
    try:
        filt = PaginatePullRequestsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    date_to_orig = filt.request.date_to  # will be +1 day in resolve_filter_prs_parameters()
    # we ignore events and stages because we cannot do anything with them
    repos, _, _, participants, labels, jira, settings, prefixer, meta_ids = \
        await resolve_filter_prs_parameters(filt.request, request)
    prefixer.cancel()
    branches, default_branches = await BranchMiner.extract_branches(
        repos, meta_ids, request.mdb, request.cache)
    # we ignore the ambiguous PRs, thus producing a pessimistic prediction (that's OK)
    done_ats, _ = await load_precomputed_done_timestamp_filters(
        filt.request.date_from, filt.request.date_to, repos, participants, labels,
        default_branches, filt.request.exclude_inactive, settings,
        filt.request.account, request.pdb)
    tasks = [
        PullRequestMiner.fetch_prs(
            filt.request.date_from, filt.request.date_to, repos, participants, labels, jira,
            filt.request.exclude_inactive, PullRequest.node_id.notin_(done_ats), None, branches,
            None, filt.request.account, meta_ids, request.mdb, request.pdb, request.cache,
            columns=[PullRequest.node_id, PullRequest.updated_at]),
    ]
    if filt.request.jira:
        tasks.append(PullRequestMiner.filter_jira(
            done_ats, jira, meta_ids, request.mdb, request.cache, columns=[PullRequest.node_id]))
        (other_prs, _), passed_jira = await gather(*tasks)
        done_ats = {k: done_ats[k] for k in passed_jira.index.values}
    else:
        other_prs, _ = await tasks[0]
    if done_ats:
        updateds = np.concatenate([other_prs[PullRequest.updated_at.key].values,
                                   np.asarray(list(done_ats.values()))]).astype("datetime64[ns]")
    elif len(other_prs) > 0:
        updateds = other_prs[PullRequest.updated_at.key].values
    else:
        updateds = np.array([filt.request.date_from, date_to_orig], dtype="datetime64[ns]")
    updateds = np.sort(updateds.astype("datetime64[D]"))
    split = updateds[::-filt.batch]
    split = np.concatenate([split, [updateds[0]]])  # append the other end
    split = np.unique(split)[::-1].tolist()  # there can be duplications
    if len(split) == 1:
        # all PRs happened on the same day
        split *= 2
    model = PullRequestPaginationPlan(updated=split)
    return model_response(model)
