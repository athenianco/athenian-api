from collections import defaultdict
from itertools import chain
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

from aiohttp import web

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.cache import expires_header, short_term_exptime
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.datetime_utils import split_to_time_intervals
from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.entries import make_calculator
from athenian.api.controllers.jira import get_jira_installation, get_jira_installation_or_none
from athenian.api.controllers.miners.access_classes import AccessChecker
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.commit import FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import DeveloperTopic
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind, \
    ReleaseParticipants, ReleaseParticipationKind
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.controllers.settings import LogicalRepositorySettings, Settings
from athenian.api.controllers.with_ import compile_developers, fetch_teams_map, \
    resolve_withgroups, scan_for_teams
from athenian.api.models.web import CalculatedCodeCheckMetrics, CalculatedCodeCheckMetricsItem, \
    CalculatedDeploymentMetric, CalculatedDeveloperMetrics, CalculatedDeveloperMetricsItem, \
    CalculatedLinearMetricValues, CalculatedPullRequestMetrics, CalculatedPullRequestMetricsItem, \
    CalculatedReleaseMetric, CodeBypassingPRsMeasurement, CodeCheckMetricsRequest, CodeFilter, \
    DeploymentMetricsRequest, DeveloperMetricsRequest, ForSetCodeChecks, ForSetDeployments, \
    ForSetDevelopers, ForSetPullRequests, PullRequestMetricID, ReleaseMetricsRequest, ReleaseWith
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.models.web.pull_request_metrics_request import PullRequestMetricsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass


@dataclass(slots=True, frozen=True)
class FilterPRs:
    """Compiled pull requests filter."""

    service: str
    repogroups: List[Set[str]]
    participants: List[Dict[PRParticipationKind, Set[str]]]
    labels: LabelFilter
    jira: JIRAFilter
    for_set_index: int
    """Index of `for_set` inside its containing sequence."""
    for_set: ForSetPullRequests
    """The original ForSetPullRequests used to compile this object."""


@dataclass(slots=True, frozen=True)
class FilterDevs:
    """Compiled developers filter."""

    service: str
    repogroups: List[Set[str]]
    developers: List[PRParticipants]
    labels: LabelFilter
    jira: JIRAFilter
    for_set: ForSetDevelopers
    """The original ForSetDevelopers used to compile this object."""


@dataclass(slots=True, frozen=True)
class FilterChecks:
    """Compiled checks filter."""

    service: str
    repogroups: List[Set[str]]
    pusher_groups: List[Sequence[str]]
    labels: LabelFilter
    jira: JIRAFilter
    for_set: ForSetCodeChecks
    """The original ForSetCodeChecks used to compile this object ."""


@dataclass(slots=True, frozen=True)
class FilterDeployments:
    """Compiled deployments filter."""

    service: str
    repogroups: List[Set[str]]
    participant_groups: List[Dict[ReleaseParticipationKind, List[int]]]
    envgroups: List[List[str]]
    with_labels: Dict[str, Any]
    without_labels: Dict[str, Any]
    pr_labels: LabelFilter
    jira: JIRAFilter
    for_set: ForSetDeployments
    """The original `ForSetDeployments` used to compile this object."""
    for_set_index: int
    """Index of `for_set` inside its containing sequence."""


@expires_header(short_term_exptime)
@weight(10)
async def calc_metrics_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate linear metrics over PRs.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    try:
        filt = PullRequestMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))

    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, filt.account)
    logical_settings = await settings.list_logical_repositories(prefixer)
    filters, repos = await compile_filters_prs(
        filt.for_, request, filt.account, meta_ids, prefixer, logical_settings)
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)

    """
    @se7entyse7en:
    It seems weird to me that the generated class constructor accepts None as param and it
    doesn't on setters. Probably it would have much more sense to generate a class that doesn't
    accept the params at all or that it does not default to None. :man_shrugging:

    @vmarkovtsev:
    This is exactly what I did the other day. That zalando/especifico thingie which glues OpenAPI
    and asyncio together constructs all the models by calling their __init__ without any args and
    then setting individual attributes. So we crash somewhere in from_dict() or to_dict() if we
    make something required.
    """
    met = CalculatedPullRequestMetrics()
    met.date_from = filt.date_from
    met.date_to = filt.date_to
    met.timezone = filt.timezone
    met.granularities = filt.granularities
    met.quantiles = filt.quantiles
    met.metrics = filt.metrics
    met.exclude_inactive = filt.exclude_inactive
    met.calculated = []

    settings = Settings.from_request(request, filt.account)
    release_settings, (branches, default_branches), account_bots = await gather(
        settings.list_release_matches(repos),
        BranchMiner.extract_branches(
            repos, prefixer, meta_ids, request.mdb, request.cache, strip=True),
        bots(filt.account, meta_ids, request.mdb, request.sdb, request.cache),
    )

    @sentry_span
    async def calculate_for_set_metrics(filter_prs: FilterPRs):
        calculator = make_calculator(
            filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
        for_set = filter_prs.for_set
        check_environments(filt.metrics, filter_prs.for_set_index, for_set)
        metric_values = await calculator.calc_pull_request_metrics_line_github(
            filt.metrics, time_intervals, filt.quantiles or (0, 1), for_set.lines or [],
            for_set.environments or [], filter_prs.repogroups, filter_prs.participants,
            filter_prs.labels, filter_prs.jira, filt.exclude_inactive, account_bots,
            release_settings, logical_settings, prefixer, branches, default_branches,
            filt.fresh,
        )
        mrange = range(len(met.metrics))
        for lines_group_index, lines_group in enumerate(metric_values):
            for repos_group_index, with_groups in enumerate(lines_group):
                for with_group_index, repos_group in enumerate(with_groups):
                    group_for_set = for_set \
                        .select_lines(lines_group_index) \
                        .select_repogroup(repos_group_index) \
                        .select_withgroup(with_group_index)
                    for granularity, ts, mvs in zip(filt.granularities,
                                                    time_intervals,
                                                    repos_group):
                        cm = CalculatedPullRequestMetricsItem(
                            for_=group_for_set,
                            granularity=granularity,
                            values=[CalculatedLinearMetricValues(
                                date=(d - tzoffset).date(),
                                values=[mvs[i][m].value for m in mrange],
                                confidence_mins=[mvs[i][m].confidence_min for m in mrange],
                                confidence_maxs=[mvs[i][m].confidence_max for m in mrange],
                                confidence_scores=[mvs[i][m].confidence_score() for m in mrange],
                            ) for i, d in enumerate(ts[:-1])])
                        for v in cm.values:
                            if sum(1 for c in v.confidence_scores if c is not None) == 0:
                                v.confidence_mins = None
                                v.confidence_maxs = None
                                v.confidence_scores = None
                        met.calculated.append(cm)
    await gather(*(calculate_for_set_metrics(filter_prs) for filter_prs in filters))
    return model_response(met)


async def compile_filters_prs(for_sets: List[ForSetPullRequests],
                              request: AthenianWebRequest,
                              account: int,
                              meta_ids: Tuple[int, ...],
                              prefixer: Prefixer,
                              logical_settings: LogicalRepositorySettings,
                              ) -> Tuple[List[FilterPRs], Set[str]]:
    """
    Build the list of filters for a given list of ForSetPullRequests-s.

    We dereference repository sets and validate access permissions.

    :param for_sets: Paired lists of repositories, developers, etc.
    :param request: Our incoming request to take the metadata DB, the user ID, the cache.
    :param account: Account ID on behalf of which we are loading reposets.
    :return: 1. Resulting list of filters; \
             2. The set of all repositories after dereferencing, with service prefixes.
    """
    filters = []
    checkers = {}
    all_repos = set()
    for i, for_set in enumerate(for_sets):
        repos, prefix, service = await _extract_repos(
            request, logical_settings, account, meta_ids, for_set.repositories,
            i, all_repos, checkers)
        if for_set.repogroups is not None:
            repogroups = [set(chain.from_iterable(repos[i] for i in group))
                          for group in for_set.repogroups]
        else:
            repogroups = [set(chain.from_iterable(repos))]
        withgroups = await resolve_withgroups(
            for_set.withgroups or ([for_set.with_] if for_set.with_ else []),
            PRParticipationKind,
            False,
            account,
            prefix,
            ".for[%d].%s" % (i, "withgroups" if i < len(for_set.withgroups or []) else "with"),
            prefixer,
            request.sdb,
            group_type=set,
        )
        labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
        jira = await _compile_jira(for_set, account, request)
        filters.append(FilterPRs(service, repogroups, withgroups, labels, jira, i, for_set))
    return filters, all_repos


def check_environments(metrics: Collection[str],
                       for_index: int,
                       for_set: ForSetPullRequests,
                       ) -> None:
    """Raise InvalidRequestError if there are deployment metrics and no environments."""
    if dep_metrics := set(metrics).intersection(
            {m for m in PullRequestMetricID if "deployment" in m}) \
            and not for_set.environments:
        raise ResponseError(InvalidRequestError(
            f".for[{for_index}].environments",
            detail=f"Metrics {dep_metrics} require setting `environments`."))


async def _compile_filters_devs(for_sets: List[ForSetDevelopers],
                                request: AthenianWebRequest,
                                account: int,
                                meta_ids: Tuple[int, ...],
                                prefixer: Prefixer,
                                logical_settings: LogicalRepositorySettings,
                                ) -> Tuple[List[FilterDevs], Set[str]]:
    """
    Build the list of filters for a given list of ForSetDevelopers'.

    We dereference repository sets and validate access permissions.

    :param for_sets: Paired lists of repositories, developers, and other selectors.
    :param request: Our incoming request to take the metadata DB, the user ID, the cache.
    :param account: Account ID on behalf of which we are loading reposets.
    :param meta_ids: Metadata (GitHub) account IDs.
    :return: Resulting list of filters and the list of all repositories after dereferencing, \
             with service prefixes.
    """
    filters = []
    checkers = {}
    all_repos = set()
    for i, for_set in enumerate(for_sets):
        repos, prefix, service = await _extract_repos(
            request, logical_settings, account, meta_ids, for_set.repositories,
            i, all_repos, checkers)
        if for_set.repogroups is not None:
            repogroups = [set(chain.from_iterable(repos[i] for i in group))
                          for group in for_set.repogroups]
        else:
            repogroups = [set(chain.from_iterable(repos))]
        devs = compile_developers(
            for_set.developers, {}, prefix, False, prefixer, f".for[{i}].developers", unique=False)
        labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
        jira = await _compile_jira(for_set, account, request)
        filters.append(FilterDevs(service, repogroups, devs, labels, jira, for_set))
    return filters, all_repos


async def compile_filters_checks(for_sets: List[ForSetCodeChecks],
                                 request: AthenianWebRequest,
                                 account: int,
                                 meta_ids: Tuple[int, ...],
                                 prefixer: Prefixer,
                                 logical_settings: LogicalRepositorySettings,
                                 ) -> List[FilterChecks]:
    """
    Build the list of filters for a given list of ForSetCodeChecks'.

    We dereference repository sets and validate access permissions.

    :param for_sets: Paired lists of repositories, commit authors, etc.
    :param request: Our incoming request to take the metadata DB, the user ID, the cache.
    :param account: Account ID on behalf of which we are loading reposets.
    :return: Resulting list of filters.
    """
    filters = []
    checkers = {}
    all_repos = set()
    for i, for_set in enumerate(for_sets):
        repos, prefix, service = await _extract_repos(
            request, logical_settings, account, meta_ids, for_set.repositories,
            i, all_repos, checkers)
        if for_set.repogroups is not None:
            repogroups = [set(chain.from_iterable(repos[i] for i in group))
                          for group in for_set.repogroups]
        else:
            repogroups = [set(chain.from_iterable(repos))]
        pusher_groups = (
            (for_set.pusher_groups or [])
            +
            ([for_set.pushers] if for_set.pushers else [])
        )
        teams = set()

        def ptr(j: int) -> str:
            return ".for[%d].%s" % (
                i,
                "pusher_groups" if j < len(for_set.pusher_groups or []) else "pushers",
            )

        for j, pushers in enumerate(pusher_groups):
            scan_for_teams(pushers, teams, ptr(j))
        teams_map = await fetch_teams_map(teams, account, request.sdb)
        commit_author_groups = []
        for j, pushers in enumerate(pusher_groups):
            if len(ca_group := compile_developers(
                    pushers, teams_map, prefix, False, prefixer, ptr(j))):
                commit_author_groups.append(sorted(ca_group))
        labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
        jira = await _compile_jira(for_set, account, request)
        filters.append(
            FilterChecks(service, repogroups, commit_author_groups, labels, jira, for_set),
        )
    return filters


async def _compile_filters_deployments(for_sets: List[ForSetDeployments],
                                       request: AthenianWebRequest,
                                       account: int,
                                       meta_ids: Tuple[int, ...],
                                       prefixer: Prefixer,
                                       logical_settings: LogicalRepositorySettings,
                                       ) -> List[FilterDeployments]:
    filters = []
    checkers = {}
    all_repos = set()
    for i, for_set in enumerate(for_sets):
        repos, prefix, service = await _extract_repos(
            request, logical_settings, account, meta_ids,
            for_set.repositories, i, all_repos, checkers)
        if for_set.repogroups is not None:
            repogroups = [set(chain.from_iterable(repos[i] for i in group))
                          for group in for_set.repogroups]
        else:
            repogroups = [set(chain.from_iterable(repos))]
        withgroups = await resolve_withgroups(
            for_set.withgroups or ([for_set.with_] if for_set.with_ else []),
            ReleaseParticipationKind,
            True,
            account,
            prefix,
            ".for[%d].%s" % (i, "withgroups" if i < len(for_set.withgroups or []) else "with"),
            prefixer,
            request.sdb,
        )
        pr_labels = LabelFilter.from_iterables(
            for_set.pr_labels_include, for_set.pr_labels_exclude)
        jira = await _compile_jira(for_set, account, request)
        if for_set.environments:
            envs = [[env] for env in for_set.environments]
        elif for_set.envgroups:
            envs = for_set.envgroups
        else:
            envs = []
        filters.append(FilterDeployments(
            service, repogroups, withgroups, envs,
            for_set.with_labels or {}, for_set.without_labels or {},
            pr_labels, jira, for_set, i,
        ))
    return filters


async def _compile_jira(for_set, account: int, request: AthenianWebRequest) -> JIRAFilter:
    try:
        return JIRAFilter.from_web(
            for_set.jira,
            await get_jira_installation(account, request.sdb, request.mdb, request.cache))
    except ResponseError:
        return JIRAFilter.empty()


async def _extract_repos(request: AthenianWebRequest,
                         logical_settings: LogicalRepositorySettings,
                         account: int,
                         meta_ids: Tuple[int, ...],
                         for_set: List[str],
                         for_set_index: int,
                         all_repos: Set[str],
                         checkers: Dict[str, AccessChecker],
                         ) -> Tuple[List[Set[str]], str, str]:
    async def login_loader() -> str:
        return (await request.user()).login

    pointer = ".for[%d].repositories" % for_set_index
    resolved, prefix = await resolve_repos(
        for_set, account, request.uid, login_loader, logical_settings, meta_ids,
        request.sdb, request.mdb, request.cache, request.app["slack"],
        strip_prefix=False, separate=True, checkers=checkers, pointer=pointer)
    all_repos.update(chain.from_iterable(resolved))
    resolved = [{r.split("/", 1)[1] for r in rs} for rs in resolved]
    # FIXME(vmarkovtsev): yeah, hardcode "github" because this is the only one we really support
    return resolved, prefix, "github"


@expires_header(short_term_exptime)
@weight(1.5)
async def calc_code_bypassing_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Measure the amount of code that was pushed outside of pull requests."""
    try:
        filt = CodeFilter.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))

    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, filt.account)
    logical_settings = await settings.list_logical_repositories(prefixer)

    async def login_loader() -> str:
        return (await request.user()).login

    repos, _ = await resolve_repos(
        filt.in_, filt.account, request.uid, login_loader, logical_settings, meta_ids,
        request.sdb, request.mdb, request.cache, request.app["slack"])
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularity, filt.timezone)
    with_author = [s.rsplit("/", 1)[1] for s in (filt.with_author or [])]
    with_committer = [s.rsplit("/", 1)[1] for s in (filt.with_committer or [])]
    calculator = make_calculator(
        filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
    stats = await calculator.calc_code_metrics_github(
        FilterCommitsProperty.BYPASSING_PRS, time_intervals, repos, with_author,
        with_committer, filt.only_default_branch, prefixer)  # type: List[CodeStats]
    model = [
        CodeBypassingPRsMeasurement(
            date=(d - tzoffset).date(),
            bypassed_commits=s.queried_number_of_commits,
            bypassed_lines=s.queried_number_of_lines,
            total_commits=s.total_number_of_commits,
            total_lines=s.total_number_of_lines,
        )
        for d, s in zip(time_intervals[:-1], stats)]
    return model_response(model)


@expires_header(short_term_exptime)
@weight(1.5)
async def calc_metrics_developers(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over developer activities."""
    try:
        filt = DeveloperMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))

    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, filt.account)
    logical_settings = await settings.list_logical_repositories(prefixer)
    filters, all_repos = await _compile_filters_devs(
        filt.for_, request, filt.account, meta_ids, prefixer, logical_settings,
    )
    if filt.date_to < filt.date_from:
        raise ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        ))
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)
    settings = Settings.from_request(request, filt.account)
    release_settings = await settings.list_release_matches(all_repos)

    met = CalculatedDeveloperMetrics()
    met.date_from = filt.date_from
    met.date_to = filt.date_to
    met.timezone = filt.timezone
    met.metrics = filt.metrics
    met.granularities = filt.granularities
    met.calculated = []
    topics = {DeveloperTopic(t) for t in filt.metrics}
    tasks = []
    for_sets = []

    for devs_filter in filters:
        if devs_filter.for_set.aggregate_devgroups:
            dev_groups = [
                [devs_filter.developers[i] for i in group]
                for group in devs_filter.for_set.aggregate_devgroups
            ]
        else:
            dev_groups = [[dev] for dev in devs_filter.developers]
        calculator = make_calculator(
            filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
        tasks.append(calculator.calc_developer_metrics_github(
            dev_groups, devs_filter.repogroups, time_intervals, topics,
            devs_filter.labels, devs_filter.jira,
            release_settings, logical_settings, prefixer))
        for_sets.append(devs_filter.for_set)
    all_stats = await gather(*tasks)
    for (stats_metrics, stats_topics), for_set in zip(all_stats, for_sets):
        topic_order = [stats_topics.index(DeveloperTopic(t)) for t in filt.metrics]
        for repogroup_index, repogroup_metrics in enumerate(stats_metrics):
            for granularity, ts, dev_metrics in zip(
                    filt.granularities, time_intervals, repogroup_metrics):
                values = []
                for ts_metrics in dev_metrics:
                    values.append(ts_values := [])
                    for date, metrics in zip(ts, ts_metrics):
                        metrics = [metrics[i] for i in topic_order]
                        confidence_mins = [m.confidence_min for m in metrics]
                        if any(confidence_mins):
                            confidence_maxs = [m.confidence_max for m in metrics]
                            confidence_scores = [m.confidence_score() for m in metrics]
                        else:
                            confidence_mins = confidence_maxs = confidence_scores = None
                        ts_values.append(CalculatedLinearMetricValues(
                            date=(date - tzoffset).date(),
                            values=[m.value for m in metrics],
                            confidence_mins=confidence_mins,
                            confidence_maxs=confidence_maxs,
                            confidence_scores=confidence_scores,
                        ))
                met.calculated.append(CalculatedDeveloperMetricsItem(
                    for_=for_set.select_repogroup(repogroup_index),
                    granularity=granularity,
                    values=values,
                ))
    return model_response(met)


async def _compile_filters_releases(request: AthenianWebRequest,
                                    for_sets: List[List[str]],
                                    with_: Optional[List[ReleaseWith]],
                                    account: int,
                                    meta_ids: Tuple[int, ...],
                                    ) -> Tuple[List[Tuple[str, str, Tuple[Set[str], List[str]]]],
                                               Set[str],
                                               Prefixer,
                                               LogicalRepositorySettings,
                                               List[ReleaseParticipants]]:
    filters = []
    checkers = {}
    all_repos = set()
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, account)
    logical_settings = await settings.list_logical_repositories(prefixer)
    for i, for_set in enumerate(for_sets):
        repos, prefix, service = await _extract_repos(
            request, logical_settings, account, meta_ids, for_set, i, all_repos, checkers)
        filters.append((service, prefix, (set(chain.from_iterable(repos)), for_set)))
    withgroups = await resolve_withgroups(
        with_,
        ReleaseParticipationKind,
        True,
        account,
        None,
        ".with",
        prefixer,
        request.sdb,
    )
    return filters, all_repos, prefixer, logical_settings, withgroups


@expires_header(short_term_exptime)
@weight(4)
async def calc_metrics_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate linear metrics over releases."""
    try:
        filt = ReleaseMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))
    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    filters, repos, prefixer, logical_settings, participants = await _compile_filters_releases(
        request, filt.for_, filt.with_, filt.account, meta_ids)
    grouped_for_sets = defaultdict(list)
    grouped_repos = defaultdict(list)
    for service, prefix, (for_set_repos, for_set) in filters:
        grouped_for_sets[service].append((prefix, for_set))
        grouped_repos[service].append(for_set_repos)
    del filters
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)

    settings = Settings.from_request(request, filt.account)
    release_settings, (branches, default_branches), jira_ids = await gather(
        settings.list_release_matches(repos),
        BranchMiner.extract_branches(
            repos, prefixer, meta_ids, request.mdb, request.cache, strip=True),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
    )
    met = []

    @sentry_span
    async def calculate_for_set_metrics(service, repos, for_sets):
        calculator = make_calculator(
            filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
        release_metric_values, release_matches = await calculator.calc_release_metrics_line_github(
            filt.metrics, time_intervals, filt.quantiles or (0, 1), repos, participants,
            LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude),
            JIRAFilter.from_web(filt.jira, jira_ids), release_settings, logical_settings, prefixer,
            branches, default_branches)
        release_matches = {k: v.name for k, v in release_matches.items()}
        mrange = range(len(filt.metrics))
        for with_, repos_mvs in zip((filt.with_ or [None]), release_metric_values):
            for (prefix, for_set), repo_group, granular_mvs in zip(for_sets, repos, repos_mvs):
                for granularity, ts, mvs in zip(filt.granularities, time_intervals, granular_mvs):
                    my_release_matches = {}
                    for r in repo_group:
                        r = prefix + r
                        try:
                            my_release_matches[r] = release_matches[r]
                        except KeyError:
                            continue
                    cm = CalculatedReleaseMetric(
                        for_=for_set,
                        with_=with_,
                        matches=my_release_matches,
                        metrics=filt.metrics,
                        granularity=granularity,
                        values=[CalculatedLinearMetricValues(
                            date=(d - tzoffset).date(),
                            values=[mvs[i][m].value for m in mrange],
                            confidence_mins=[mvs[i][m].confidence_min for m in mrange],
                            confidence_maxs=[mvs[i][m].confidence_max for m in mrange],
                            confidence_scores=[mvs[i][m].confidence_score() for m in mrange],
                        ) for i, d in enumerate(ts[:-1])])
                    for v in cm.values:
                        if sum(1 for c in v.confidence_scores if c is not None) == 0:
                            v.confidence_mins = None
                            v.confidence_maxs = None
                            v.confidence_scores = None
                    met.append(cm)

    await gather(*(
        calculate_for_set_metrics(service, repos, grouped_for_sets[service])
        for service, repos in grouped_repos.items()
    ))
    return model_response(met)


@expires_header(short_term_exptime)
@weight(1)
async def calc_metrics_code_checks(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics on continuous integration runs, such as GitHub Actions, Jenkins, Circle, \
    etc."""
    try:
        filt = CodeCheckMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))

    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, filt.account)
    logical_settings = await settings.list_logical_repositories(prefixer)
    filters = await compile_filters_checks(
        filt.for_, request, filt.account, meta_ids, prefixer, logical_settings)
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)
    met = CalculatedCodeCheckMetrics()
    met.date_from = filt.date_from
    met.date_to = filt.date_to
    met.timezone = filt.timezone
    met.granularities = filt.granularities
    met.quantiles = filt.quantiles
    met.metrics = filt.metrics
    met.split_by_check_runs = filt.split_by_check_runs
    met.calculated = []

    @sentry_span
    async def calculate_for_set_metrics(filter_checks: FilterChecks):
        for_set = filter_checks.for_set
        calculator = make_calculator(
            filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
        metric_values, group_suite_counts, suite_sizes = \
            await calculator.calc_check_run_metrics_line_github(
                filt.metrics, time_intervals, filt.quantiles or (0, 1),
                filter_checks.repogroups, filter_checks.pusher_groups,
                filt.split_by_check_runs, filter_checks.labels, filter_checks.jira,
                for_set.lines or [], logical_settings)
        mrange = range(len(met.metrics))
        for pushers_group_index, pushers_group in enumerate(metric_values):
            for repos_group_index, repos_group in enumerate(pushers_group):
                for lines_group_index, lines_group in enumerate(repos_group):
                    my_suite_counts = group_suite_counts[
                        pushers_group_index, repos_group_index, lines_group_index]
                    total_group_suites = my_suite_counts.sum()
                    for suite_size_group_index, suite_size_group in enumerate(lines_group):
                        group_for_set = for_set \
                            .select_pushers_group(pushers_group_index) \
                            .select_repogroup(repos_group_index) \
                            .select_lines(lines_group_index)  # type: ForSetCodeChecks
                        if filt.split_by_check_runs:
                            suite_size = suite_sizes[suite_size_group_index]
                            group_suites_count_ratio = \
                                my_suite_counts[suite_size_group_index] / total_group_suites
                        else:
                            suite_size = group_suites_count_ratio = None
                        for granularity, ts, mvs in zip(
                                filt.granularities, time_intervals, suite_size_group):
                            cm = CalculatedCodeCheckMetricsItem(
                                for_=group_for_set,
                                granularity=granularity,
                                check_runs=suite_size,
                                suites_ratio=group_suites_count_ratio,
                                values=[CalculatedLinearMetricValues(
                                    date=(d - tzoffset).date(),
                                    values=[mvs[i][m].value for m in mrange],
                                    confidence_mins=[mvs[i][m].confidence_min for m in mrange],
                                    confidence_maxs=[mvs[i][m].confidence_max for m in mrange],
                                    confidence_scores=[
                                        mvs[i][m].confidence_score() for m in mrange
                                    ],
                                ) for i, d in enumerate(ts[:-1])])
                            for v in cm.values:
                                if sum(1 for c in v.confidence_scores if c is not None) == 0:
                                    v.confidence_mins = None
                                    v.confidence_maxs = None
                                    v.confidence_scores = None
                            met.calculated.append(cm)

    await gather(*(calculate_for_set_metrics(filter_checks) for filter_checks in filters))
    return model_response(met)


@expires_header(short_term_exptime)
@weight(2)
async def calc_metrics_deployments(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics on deployments submitted by `/events/deployments`."""
    try:
        filt = DeploymentMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))
    meta_ids, jira_ids = await gather(
        get_metadata_account_ids(filt.account, request.sdb, request.cache),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
    )
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, filt.account)
    logical_settings = await settings.list_logical_repositories(prefixer)

    filters = await _compile_filters_deployments(
        filt.for_, request, filt.account, meta_ids, prefixer, logical_settings,
    )
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)
    calculated = []
    release_settings, (branches, default_branches) = \
        await gather(
            Settings.from_request(request, filt.account).list_release_matches(),  # no "repos"!
            BranchMiner.extract_branches(None, prefixer, meta_ids, request.mdb, request.cache),
    )

    @sentry_span
    async def calculate_for_set_metrics(filter_deployments: FilterDeployments):
        for_set = filter_deployments.for_set
        calculator = make_calculator(
            filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
        metric_values = await calculator.calc_deployment_metrics_line_github(
            filt.metrics, time_intervals, filt.quantiles or (0, 1),
            filter_deployments.repogroups, filter_deployments.participant_groups,
            filter_deployments.envgroups, filter_deployments.pr_labels,
            filter_deployments.with_labels, filter_deployments.without_labels,
            filter_deployments.jira, release_settings, logical_settings, prefixer,
            branches, default_branches, jira_ids,
        )
        mrange = range(len(filt.metrics))
        for with_group_index, with_group in enumerate(metric_values):
            for repos_group_index, repos_group in enumerate(with_group):
                for env_index, env_group in enumerate(repos_group):
                    group_for_set = for_set \
                        .select_withgroup(with_group_index) \
                        .select_repogroup(repos_group_index) \
                        .select_envgroup(env_index)  # type: ForSetDeployments
                    for granularity, ts, mvs in zip(
                            filt.granularities, time_intervals, env_group):
                        cm = CalculatedDeploymentMetric(
                            for_=group_for_set,
                            metrics=filt.metrics,
                            granularity=granularity,
                            values=[CalculatedLinearMetricValues(
                                date=(d - tzoffset).date(),
                                values=[mvs[i][m].value for m in mrange],
                                confidence_mins=[mvs[i][m].confidence_min for m in mrange],
                                confidence_maxs=[mvs[i][m].confidence_max for m in mrange],
                                confidence_scores=[mvs[i][m].confidence_score() for m in mrange],
                            ) for i, d in enumerate(ts[:-1])])
                        for v in cm.values:
                            if sum(1 for c in v.confidence_scores if c is not None) == 0:
                                v.confidence_mins = None
                                v.confidence_maxs = None
                                v.confidence_scores = None
                        calculated.append(cm)

    await gather(*(calculate_for_set_metrics(filter_) for filter_ in filters))
    return model_response(calculated)
