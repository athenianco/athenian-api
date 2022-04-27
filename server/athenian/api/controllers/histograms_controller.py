from collections import defaultdict

from aiohttp import web

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.cache import expires_header, short_term_exptime
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.features.entries import UnsupportedMetricError
from athenian.api.controllers.features.histogram import HistogramParameters, Scale
from athenian.api.controllers.metrics_controller import check_environments, \
    compile_filters_checks, compile_filters_prs, FilterPRs, get_calculators_for_request
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import Settings
from athenian.api.models.web import BadRequestError, CalculatedCodeCheckHistogram, \
    CalculatedPullRequestHistogram, CodeCheckHistogramsRequest, ForSetCodeChecks, Interquartile, \
    InvalidRequestError, PullRequestHistogramsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@expires_header(short_term_exptime)
@weight(10)
async def calc_histogram_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over PR distributions."""
    try:
        filt = PullRequestHistogramsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))
    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, filt.account)
    logical_settings = await settings.list_logical_repositories(prefixer)

    filters, repos = await compile_filters_prs(
        filt.for_, request, filt.account, meta_ids, prefixer, logical_settings,
    )
    time_from, time_to = filt.resolve_time_from_and_to()
    release_settings, (branches, default_branches), account_bots, calculators = await gather(
        Settings.from_request(request, filt.account).list_release_matches(repos),
        BranchMiner.extract_branches(
            repos, prefixer, meta_ids, request.mdb, request.cache, strip=True),
        bots(filt.account, meta_ids, request.mdb, request.sdb, request.cache),
        get_calculators_for_request({s.service for s in filters}, filt.account, meta_ids, request),
    )
    result = []

    async def calculate_for_set_histograms(filter_prs: FilterPRs):
        for_set = filter_prs.for_set
        check_environments([h.metric for h in filt.histograms], filter_prs.for_set_index, for_set)
        if for_set.environments is not None:
            if len(for_set.environments) > 1:
                raise ResponseError(InvalidRequestError(
                    f".for[{filter_prs.for_set_index}].environments",
                    "`environments` cannot contain more than one item to calculate histograms"),
                )
            environment = for_set.environments[0]
        else:
            environment = None
        defs = defaultdict(list)
        for h in filt.histograms:
            defs[HistogramParameters(
                scale=Scale[h.scale.upper()] if h.scale is not None else None,
                bins=h.bins,
                ticks=tuple(h.ticks) if h.ticks is not None else None,
            )].append(h.metric)
        calculator = calculators[filter_prs.service]
        try:
            histograms = await calculator.calc_pull_request_histograms_github(
                defs, time_from, time_to, filt.quantiles or (0, 1), for_set.lines or [],
                environment, filter_prs.repogroups, filter_prs.participants,
                filter_prs.labels, filter_prs.jira, filt.exclude_inactive,
                account_bots, release_settings, logical_settings, prefixer,
                branches, default_branches, fresh=filt.fresh)
        except UnsupportedMetricError as e:
            raise ResponseError(BadRequestError(str(e))) from None
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

    await gather(*(calculate_for_set_histograms(filter_prs) for filter_prs in filters))
    return model_response(result)


@expires_header(short_term_exptime)
@weight(1)
async def calc_histogram_code_checks(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms on continuous integration runs, such as GitHub Actions, Jenkins, \
    Circle, etc."""
    try:
        filt = CodeCheckHistogramsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))

    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, filt.account)
    logical_settings = await settings.list_logical_repositories(prefixer)

    filters = await compile_filters_checks(
        filt.for_, request, filt.account, meta_ids, logical_settings,
    )
    time_from, time_to = filt.resolve_time_from_and_to()
    calculators = await get_calculators_for_request(
        {s for s, _ in filters}, filt.account, meta_ids, request)
    result = []

    async def calculate_for_set_histograms(service, repos, pusher_groups, labels, jira, for_set):
        defs = defaultdict(list)
        for h in filt.histograms:
            defs[HistogramParameters(
                scale=Scale[h.scale.upper()] if h.scale is not None else None,
                bins=h.bins,
                ticks=tuple(h.ticks) if h.ticks is not None else None,
            )].append(h.metric)
        calculator = calculators[service]
        try:
            histograms, group_suite_counts, suite_sizes = \
                await calculator.calc_check_run_histograms_line_github(
                    defs, time_from, time_to, filt.quantiles or (0, 1), repos, pusher_groups,
                    filt.split_by_check_runs, labels, jira, for_set.lines or [], logical_settings)
        except UnsupportedMetricError as e:
            raise ResponseError(BadRequestError(str(e))) from None
        for pusher_groups in histograms:
            for pushers_group_index, pushers_group in enumerate(pusher_groups):
                for repos_group_index, repos_group in enumerate(pushers_group):
                    for lines_index, lines_group in enumerate(repos_group):
                        my_suite_counts = group_suite_counts[
                            pushers_group_index, repos_group_index, lines_index]
                        total_group_suites = my_suite_counts.sum()
                        for suite_size_group_index, suite_size_group in enumerate(lines_group):
                            group_for_set = for_set \
                                .select_pushers_group(pushers_group_index) \
                                .select_repogroup(repos_group_index) \
                                .select_lines(lines_index)  # type: ForSetCodeChecks
                            if filt.split_by_check_runs:
                                suite_size = suite_sizes[suite_size_group_index]
                                group_suites_count_ratio = \
                                    my_suite_counts[suite_size_group_index] / total_group_suites
                            else:
                                suite_size = group_suites_count_ratio = None
                            for metric, histogram in sorted(suite_size_group):
                                result.append(CalculatedCodeCheckHistogram(
                                    for_=group_for_set,
                                    check_runs=suite_size,
                                    suites_ratio=group_suites_count_ratio,
                                    metric=metric,
                                    scale=histogram.scale.name.lower(),
                                    ticks=histogram.ticks,
                                    frequencies=histogram.frequencies,
                                    interquartile=Interquartile(*histogram.interquartile),
                                ))

    tasks = [
        calculate_for_set_histograms(service, *args) for service, args in filters
    ]
    await gather(*tasks)
    return model_response(result)
