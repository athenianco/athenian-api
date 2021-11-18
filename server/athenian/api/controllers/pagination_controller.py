from aiohttp import web
import numpy as np

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_user_account_status
from athenian.api.controllers.filter_controller import resolve_filter_prs_parameters
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.precomputed_prs import DonePRFactsLoader
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
    await get_user_account_status(
        request.uid, filt.request.account, request.sdb, request.cache)
    # we ignore events and stages because we cannot do anything with them
    time_from, time_to, repos, _, _, participants, labels, jira, release_settings, \
        logical_settings, prefixer, meta_ids = \
        await resolve_filter_prs_parameters(filt.request, request)
    branches, default_branches = await BranchMiner.extract_branches(
        repos, meta_ids, request.mdb, request.cache)
    # we ignore the ambiguous PRs, thus producing a pessimistic prediction (that's OK)
    done_ats, _ = await DonePRFactsLoader.load_precomputed_done_timestamp_filters(
        time_from, time_to, repos, participants, labels,
        default_branches, filt.request.exclude_inactive, release_settings, prefixer,
        filt.request.account, request.pdb)
    done_node_ids = {}
    for node_id, repo in done_ats:
        done_node_ids.setdefault(node_id, []).append(repo)
    tasks = [
        PullRequestMiner.fetch_prs(
            time_from, time_to, repos, participants, labels, jira,
            filt.request.exclude_inactive, PullRequest.node_id.notin_(done_node_ids),
            None, branches, None, filt.request.account, meta_ids,
            request.mdb, request.pdb, request.cache,
            columns=[PullRequest.node_id, PullRequest.updated_at]),
    ]
    if filt.request.jira:
        tasks.append(PullRequestMiner.filter_jira(
            done_node_ids, jira, meta_ids, request.mdb, request.cache,
            columns=[PullRequest.node_id]))
        (other_prs, *_), passed_jira = await gather(*tasks)
        done_ats = {
            (k, r): done_ats[k, r]
            for k in passed_jira.index.values
            for r in done_node_ids[k]
        }
    else:
        other_prs, *_ = await tasks[0]
    if done_ats:
        updateds = np.concatenate([other_prs[PullRequest.updated_at.name].values,
                                   np.asarray(list(done_ats.values()))]).astype("datetime64[ns]")
    elif len(other_prs) > 0:
        updateds = other_prs[PullRequest.updated_at.name].values
    else:
        updateds = np.array([time_from, filt.request.date_to], dtype="datetime64[ns]")
    updateds = np.sort(updateds.astype("datetime64[D]"))
    split = updateds[::-filt.batch]
    split = np.concatenate([split, [updateds[0]]])  # append the other end
    split = np.unique(split)[::-1].tolist()  # there can be duplications
    if len(split) == 1:
        # all PRs happened on the same day
        split *= 2
    model = PullRequestPaginationPlan(updated=split)
    return model_response(model)
