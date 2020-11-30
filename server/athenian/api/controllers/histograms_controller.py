from collections import defaultdict

from aiohttp import web

from athenian.api.async_utils import gather
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.features.entries import METRIC_ENTRIES
from athenian.api.controllers.features.histogram import HistogramParameters, Scale
from athenian.api.controllers.metrics_controller import compile_repos_and_devs_prs
from athenian.api.controllers.settings import Settings
from athenian.api.models.web import InvalidRequestError
from athenian.api.models.web.calculated_pull_request_histogram import \
    CalculatedPullRequestHistogram, Interquartile
from athenian.api.models.web.pull_request_histograms_request import PullRequestHistogramsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


async def calc_histogram_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over PR distributions."""
    try:
        filt = PullRequestHistogramsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    filters, repos = await compile_repos_and_devs_prs(filt.for_, request, filt.account, meta_ids)
    time_from, time_to = filt.resolve_time_from_and_to()
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    result = []

    async def calculate_for_set_histograms(service, repos, devs, labels, jira, for_set):
        # for each filter, we find the functions to calculate the histograms
        defs = defaultdict(list)
        for h in filt.histograms:
            defs[HistogramParameters(
                scale=Scale[h.scale.upper()] if h.scale is not None else None,
                bins=h.bins,
                ticks=tuple(h.ticks) if h.ticks is not None else None,
            )].append(h.metric)
        try:
            group_histograms = await METRIC_ENTRIES[service]["prs_histogram"](
                meta_ids, defs, time_from, time_to, filt.quantiles or (0, 1), for_set.lines or [],
                repos, devs, labels, jira, filt.exclude_inactive, release_settings,
                filt.fresh, request.mdb, request.pdb, request.cache)
        except ValueError as e:
            raise ResponseError(InvalidRequestError(str(e))) from None
        line_index = 0
        line_bins = len(for_set.lines or [None] * 2) - 1
        assert len(group_histograms) == len(repos) * line_bins
        for group, histograms in enumerate(group_histograms):
            group //= line_bins
            group_for_set = for_set.select_lines(line_index).select_repogroup(group)
            line_index = (line_index + 1) % line_bins
            for metric, histogram in sorted(histograms):
                result.append(CalculatedPullRequestHistogram(
                    for_=group_for_set,
                    metric=metric,
                    scale=histogram.scale.name.lower(),
                    ticks=histogram.ticks,
                    frequencies=histogram.frequencies,
                    interquartile=Interquartile(*histogram.interquartile),
                ))

    tasks = []
    for service, (repos, devs, labels, jira, for_set) in filters:
        tasks.append(calculate_for_set_histograms(service, repos, devs, labels, jira, for_set))
    await gather(*tasks)
    return model_response(result)
