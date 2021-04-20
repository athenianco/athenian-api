from collections import defaultdict

from aiohttp import web

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.calculator_selector import get_calculator_for_user
from athenian.api.controllers.features.histogram import HistogramParameters, Scale
from athenian.api.controllers.metrics_controller import compile_repos_and_devs_prs
from athenian.api.controllers.settings import Settings
from athenian.api.models.web import InvalidRequestError
from athenian.api.models.web.calculated_pull_request_histogram import \
    CalculatedPullRequestHistogram, Interquartile
from athenian.api.models.web.pull_request_histograms_request import PullRequestHistogramsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@weight(10)
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
    services = set(s for s, _ in filters)
    release_settings, calculators = await gather(
        Settings.from_request(request, filt.account).list_release_matches(repos),
        get_calculator_for_user(
            services, filt.account, meta_ids, request.uid, getattr(request, "god_id", None),
            request.sdb, request.mdb, request.pdb, request.rdb, request.cache,
        ),
    )
    result = []

    async def calculate_for_set_histograms(service, repos, withgroups, labels, jira, for_set):
        # for each filter, we find the functions to calculate the histograms
        defs = defaultdict(list)
        for h in filt.histograms:
            defs[HistogramParameters(
                scale=Scale[h.scale.upper()] if h.scale is not None else None,
                bins=h.bins,
                ticks=tuple(h.ticks) if h.ticks is not None else None,
            )].append(h.metric)
        calculator = calculators[service]
        try:
            histograms = await calculator.calc_pull_request_histograms_github(
                defs, time_from, time_to, filt.quantiles or (0, 1), for_set.lines or [],
                repos, withgroups, labels, jira, filt.exclude_inactive, release_settings,
                filt.fresh)
        except ValueError as e:
            raise ResponseError(InvalidRequestError(str(e))) from None
        for line_groups in histograms:
            for line_group_index, repo_groups in enumerate(line_groups):
                for repo_group_index, with_groups in enumerate(repo_groups):
                    for with_group_index, repo_histograms in enumerate(with_groups):
                        group_for_set = for_set \
                            .select_lines(line_group_index) \
                            .select_repogroup(repo_group_index) \
                            .select_withgroup(with_group_index)
                        for metric, histogram in sorted(repo_histograms):
                            result.append(CalculatedPullRequestHistogram(
                                for_=group_for_set,
                                metric=metric,
                                scale=histogram.scale.name.lower(),
                                ticks=histogram.ticks,
                                frequencies=histogram.frequencies,
                                interquartile=Interquartile(*histogram.interquartile),
                            ))

    tasks = []
    for service, (repos, withgroups, labels, jira, for_set) in filters:
        tasks.append(calculate_for_set_histograms(
            service, repos, withgroups, labels, jira, for_set))
    await gather(*tasks)
    return model_response(result)
