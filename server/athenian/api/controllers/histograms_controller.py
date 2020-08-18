import asyncio
from collections import defaultdict
from typing import List

from aiohttp import web

from athenian.api import ResponseError
from athenian.api.controllers.features.entries import METRIC_ENTRIES
from athenian.api.controllers.features.histogram import Histogram, Scale
from athenian.api.controllers.metrics_controller import compile_repos_and_devs_prs, \
    resolve_time_from_and_to
from athenian.api.controllers.settings import Settings
from athenian.api.models.web import HistogramScale, InvalidRequestError
from athenian.api.models.web.calculated_pull_request_histogram import \
    CalculatedPullRequestHistogram
from athenian.api.models.web.pull_request_histograms_request import PullRequestHistogramsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response


async def calc_histogram_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over PR distributions."""
    try:
        filt = PullRequestHistogramsRequest.from_dict(body)  # type: PullRequestHistogramsRequest
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    filters, repos = await compile_repos_and_devs_prs(filt.for_, request, filt.account)
    time_from, time_to = resolve_time_from_and_to(filt)
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    result = []

    async def calculate_for_set_histograms(service, repos, devs, labels_include, for_set):
        calcs = defaultdict(list)
        # for each filter, we find the functions to calculate the histograms
        entries = METRIC_ENTRIES[service]["prs_histogram"]
        for m in filt.metrics:
            try:
                calcs[entries[m]].append(m)
            except KeyError:
                raise ResponseError(InvalidRequestError(
                    "Metric is not supported.", detail=m,
                )) from None
        # for each topic, we find the function to calculate the histogram and call it
        tasks = []
        for func, metrics in calcs.items():
            tasks.append(func(
                metrics, Scale[filt.scale.upper()], filt.bins or 0, time_from, time_to,
                filt.quantiles or (0, 1), repos, devs, labels_include, filt.exclude_inactive,
                release_settings, request.mdb, request.pdb, request.cache))
        if len(tasks) == 1:
            all_histograms = [await tasks[0]]  # type: List[Histogram]
        else:
            all_histograms = await asyncio.gather(*tasks, return_exceptions=True)  # type: List[Histogram]  # noqa
        for metrics, histograms in zip(calcs.values(), all_histograms):
            if isinstance(histograms, Exception):
                if isinstance(histograms, ValueError) and filt.scale == HistogramScale.LOG:
                    raise ResponseError(InvalidRequestError(
                        "Logarithmic histogram scale is incompatible with non-positive samples.",
                        detail=str(histograms),
                    )) from None
                raise histograms from None
            for metric, histogram in zip(metrics, histograms):
                result.append(CalculatedPullRequestHistogram(
                    for_=for_set,
                    metric=metric,
                    scale=histogram.scale.name.lower(),
                    ticks=histogram.ticks,
                    frequencies=histogram.frequencies,
                ))

    tasks = []
    for service, (repos, devs, labels_include, for_set) in filters:
        tasks.append(calculate_for_set_histograms(service, repos, devs, labels_include, for_set))
    if len(tasks) == 0:
        await tasks[0]
    else:
        for err in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(err, Exception):
                raise err from None
    return model_response(result)
