import asyncio
from collections import defaultdict

from aiohttp import web

from athenian.api import ResponseError
from athenian.api.controllers.features.entries import METRIC_ENTRIES
from athenian.api.controllers.features.histogram import HistogramParameters, Scale
from athenian.api.controllers.metrics_controller import compile_repos_and_devs_prs
from athenian.api.controllers.settings import Settings
from athenian.api.models.web import InvalidRequestError
from athenian.api.models.web.calculated_pull_request_histogram import \
    CalculatedPullRequestHistogram
from athenian.api.models.web.pull_request_histograms_request import PullRequestHistogramsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response


async def calc_histogram_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over PR distributions."""
    try:
        filt = PullRequestHistogramsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    filters, repos = await compile_repos_and_devs_prs(filt.for_, request, filt.account)
    time_from, time_to = filt.resolve_time_from_and_to()
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    result = []

    async def calculate_for_set_histograms(service, repos, devs, labels, jira, for_set):
        # for each filter, we find the functions to calculate the histograms
        defs = defaultdict(list)
        for h in (filt.histograms or []):
            defs[HistogramParameters(
                scale=Scale[h.scale.upper()] if h.scale is not None else None,
                bins=h.bins,
                ticks=tuple(h.ticks) if h.ticks is not None else None,
            )].append(h.metric)
        # FIXME(vmarkovtsev): this is deprecated and should be removed
        for m in (filt.metrics or []):
            defs[HistogramParameters(
                scale=Scale[filt.scale.upper()] if filt.scale is not None else Scale.LINEAR,
                bins=filt.bins or 0,
                ticks=None,
            )].append(m)
        try:
            group_histograms = await METRIC_ENTRIES[service]["prs_histogram"](
                defs, time_from, time_to, filt.quantiles or (0, 1), repos, devs, labels, jira,
                filt.exclude_inactive, release_settings, filt.fresh, request.mdb, request.pdb,
                request.cache)
        except ValueError as e:
            raise ResponseError(InvalidRequestError(str(e))) from None
        assert len(group_histograms) == len(repos)
        for group, histograms in enumerate(group_histograms):
            for metric, histogram in sorted(histograms):
                result.append(CalculatedPullRequestHistogram(
                    for_=for_set.select_repogroup(group),
                    metric=metric,
                    scale=histogram.scale.name.lower(),
                    ticks=histogram.ticks,
                    frequencies=histogram.frequencies,
                    interquartile=histogram.interquartile,
                ))

    tasks = []
    for service, (repos, devs, labels, jira, for_set) in filters:
        tasks.append(calculate_for_set_histograms(service, repos, devs, labels, jira, for_set))
    if len(tasks) == 0:
        await tasks[0]
    else:
        for err in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(err, Exception):
                raise err from None
    return model_response(result)
