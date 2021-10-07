from collections import defaultdict
from itertools import chain
from typing import Collection, Dict, Iterable, List, Sequence, Set, Tuple, Union

from aiohttp import web
import databases.core

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.connexion import ADJUST_LOAD_VAR_NAME
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.calculator_selector import get_calculators_for_account
from athenian.api.controllers.datetime_utils import split_to_time_intervals
from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.entries import MetricEntriesCalculator
from athenian.api.controllers.jira import get_jira_installation, get_jira_installation_or_none
from athenian.api.controllers.miners.access_classes import access_classes, AccessChecker
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.commit import FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import DeveloperTopic
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.release import extract_release_participants
from athenian.api.controllers.reposet import resolve_repos, resolve_reposet
from athenian.api.controllers.settings import Settings
from athenian.api.controllers.user import MANNEQUIN_PREFIX
from athenian.api.models.web import CalculatedCodeCheckMetrics, CalculatedCodeCheckMetricsItem, \
    CalculatedDeveloperMetrics, CalculatedDeveloperMetricsItem, CalculatedLinearMetricValues, \
    CalculatedPullRequestMetrics, CalculatedPullRequestMetricsItem, CalculatedReleaseMetric, \
    CodeBypassingPRsMeasurement, CodeCheckMetricsRequest, CodeFilter, DeveloperMetricsRequest, \
    ForbiddenError, ForSet, ForSetCodeChecks, ForSetDevelopers, PullRequestMetricID, \
    ReleaseMetricsRequest
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.models.web.pull_request_metrics_request import PullRequestMetricsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span

#               service                          developers                                    originals  # noqa
FilterPRs = Tuple[str, Tuple[List[Set[str]], List[PRParticipants], LabelFilter, JIRAFilter, int, ForSet]]  # noqa
#                             repositories                                              for's index

#                service                     developers
FilterDevs = Tuple[str, Tuple[List[Set[str]], List[str], ForSetDevelopers]]
#                              repositories                  originals

#                  service                      pusher groups
FilterChecks = Tuple[str, Tuple[List[Set[str]], List[List[str]], LabelFilter, JIRAFilter, ForSetCodeChecks]]  # noqa
#                               repositories                                                  originals       # noqa


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
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    filters, repos = await compile_filters_prs(filt.for_, request, filt.account, meta_ids)
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)

    """
    @se7entyse7en:
    It seems weird to me that the generated class constructor accepts None as param and it
    doesn't on setters. Probably it would have much more sense to generate a class that doesn't
    accept the params at all or that it does not default to None. :man_shrugging:

    @vmarkovtsev:
    This is exactly what I did the other day. That zalando/connexion thingie which glues OpenAPI
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

    release_settings, calculators = await gather(
        Settings.from_request(request, filt.account).list_release_matches(repos),
        get_calculators_for_request({s for s, _ in filters}, filt.account, meta_ids, request),
    )

    @sentry_span
    async def calculate_for_set_metrics(
            service, repos, withgroups, labels, jira, for_index, for_set):
        calculator = calculators[service]
        check_environments(filt.metrics, for_index, for_set)
        metric_values = await calculator.calc_pull_request_metrics_line_github(
            filt.metrics, time_intervals, filt.quantiles or (0, 1),
            for_set.lines or [], for_set.environments or [], repos, withgroups, labels, jira,
            filt.exclude_inactive, release_settings, prefixer, filt.fresh)
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
    await gather(*(
        calculate_for_set_metrics(service, repos, withgroups, labels, jira, for_index, for_set)
        for service, (repos, withgroups, labels, jira, for_index, for_set) in filters
    ))
    return model_response(met)


async def get_calculators_for_request(services: Iterable[str],
                                      account: int,
                                      meta_ids: Tuple[int, ...],
                                      request: AthenianWebRequest,
                                      ) -> Dict[str, MetricEntriesCalculator]:
    """Get the metrics calculator species for the given account."""
    calcs = await get_calculators_for_account(
        services, account, meta_ids, getattr(request, "god_id", None),
        request.sdb, request.mdb, request.pdb, request.rdb, request.cache,
        instrument=request.app["metrics_calculator"].get(),
    )
    load_delta = max(calc.load_delta for calc in calcs.values())
    request.app[ADJUST_LOAD_VAR_NAME].get()(load_delta)
    return calcs


async def compile_filters_prs(for_sets: List[ForSet],
                              request: AthenianWebRequest,
                              account: int,
                              meta_ids: Tuple[int, ...],
                              ) -> Tuple[List[FilterPRs], Set[str]]:
    """
    Build the list of filters for a given list of ForSet-s.

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
    async with request.sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos, prefix, service = await _extract_repos(
                request, account, meta_ids, for_set.repositories, i, all_repos, checkers, sdb_conn)
            if for_set.repogroups is not None:
                repogroups = [set(chain.from_iterable(repos[i] for i in group))
                              for group in for_set.repogroups]
            else:
                repogroups = [set(chain.from_iterable(repos))]
            withgroups = []
            for with_ in (for_set.withgroups or []) + ([for_set.with_] if for_set.with_ else []):
                withgroup = {}
                for k, v in with_.items():
                    if not v:
                        continue
                    withgroup[PRParticipationKind[k.upper()]] = _compile_dev_logins(
                        v, prefix, ".for[%d].%s" % (
                            i, "withgroups" if i < len(for_set.withgroups or []) else "with"))
                if withgroup:
                    withgroups.append(withgroup)
            labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
            jira = await _compile_jira(for_set, account, request)
            filters.append((service, (repogroups, withgroups, labels, jira, i, for_set)))
    return filters, all_repos


def check_environments(metrics: Collection[str], for_index: int, for_set: ForSet) -> None:
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
                                ) -> (List[FilterDevs], List[str]):
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
    async with request.sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos, prefix, service = await _extract_repos(
                request, account, meta_ids, for_set.repositories, i, all_repos, checkers, sdb_conn)
            if for_set.repogroups is not None:
                repogroups = [set(chain.from_iterable(repos[i] for i in group))
                              for group in for_set.repogroups]
            else:
                repogroups = [set(chain.from_iterable(repos))]
            devs = _compile_dev_logins(
                for_set.developers, prefix, ".for[%d].developers" % i, unique=False)
            labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
            jira = await _compile_jira(for_set, account, request)
            filters.append((service, (repogroups, devs, labels, jira, for_set)))
    return filters, all_repos


async def compile_filters_checks(for_sets: List[ForSetCodeChecks],
                                 request: AthenianWebRequest,
                                 account: int,
                                 meta_ids: Tuple[int, ...],
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
    async with request.sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos, prefix, service = await _extract_repos(
                request, account, meta_ids, for_set.repositories, i, all_repos, checkers, sdb_conn)
            if for_set.repogroups is not None:
                repogroups = [set(chain.from_iterable(repos[i] for i in group))
                              for group in for_set.repogroups]
            else:
                repogroups = [set(chain.from_iterable(repos))]
            commit_author_groups = []
            for pushers in ((for_set.pusher_groups or []) +
                            ([for_set.pushers] if for_set.pushers else [])):
                if ca_group := _compile_dev_logins(pushers, prefix, ".for[%d].%s" % (
                        i, "commit_author_groups"
                        if i < len(for_set.pusher_groups or []) else "pushers")):
                    commit_author_groups.append(sorted(ca_group))
            labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
            jira = await _compile_jira(for_set, account, request)
            filters.append((service, (repogroups, commit_author_groups, labels, jira, for_set)))
    return filters


def _compile_dev_logins(developers: Iterable[str],
                        prefix: str,
                        pointer: str,
                        unique: bool = True) -> Union[List[str], Set[str]]:
    devs = []
    prefix = prefix.rstrip("/")
    for i, dev in enumerate(developers):
        parts = dev.split("/")
        dev_prefix, dev_login = parts[0], parts[-1]
        if dev_prefix != prefix and dev_prefix != MANNEQUIN_PREFIX:
            raise ResponseError(InvalidRequestError(
                detail='User "%s" has prefix "%s" that does not match with the repository prefix '
                       '"%s"' % (dev, dev_prefix, prefix),
                pointer=pointer + "[%d]" % i,
            ))
        devs.append(dev_login)
    if unique:
        return set(devs)
    return devs


async def _compile_jira(for_set, account: int, request: AthenianWebRequest) -> JIRAFilter:
    try:
        return JIRAFilter.from_web(
            for_set.jira,
            await get_jira_installation(account, request.sdb, request.mdb, request.cache))
    except ResponseError:
        return JIRAFilter.empty()


async def _extract_repos(request: AthenianWebRequest,
                         account: int,
                         meta_ids: Tuple[int, ...],
                         for_set: List[str],
                         for_set_index: int,
                         all_repos: Set[str],
                         checkers: Dict[str, AccessChecker],
                         sdb: databases.core.Connection,
                         ) -> Tuple[Sequence[Set[str]], str, str]:
    user = request.uid
    prefix = None
    resolved = await gather(*[
        resolve_reposet(r, ".for[%d].repositories[%d]" % (
            for_set_index, j), user, account, sdb, request.cache)
        for j, r in enumerate(for_set)], op="resolve_reposet-s")
    for repos in resolved:
        for i, repo in enumerate(repos):
            repo_prefix, repos[i] = repo.split("/", 1)
            if prefix is None:
                prefix = repo_prefix
            elif prefix != repo_prefix:
                raise ResponseError(InvalidRequestError(
                    detail='mixed providers are not allowed in the same "for" element',
                    pointer=".for[%d].repositories" % for_set_index,
                ))
            all_repos.add(repo)
    if prefix is None:
        raise ResponseError(InvalidRequestError(
            detail='the provider of a "for" element is unsupported or the set is empty',
            pointer=".for[%d].repositories" % for_set_index,
        ))
    service = "github"  # hardcode "github" because we do not really support others
    if (checker := checkers.get(service)) is None:
        checkers[service] = (checker := await access_classes[service](
            account, meta_ids, sdb, request.mdb, request.cache,
        ).load())
    if denied := await checker.check(set(chain.from_iterable(resolved))):
        raise ResponseError(ForbiddenError(
            detail="some repositories in .for[%d].repositories are access denied on %s: %s" % (
                for_set_index, service, denied),
        ))
    return resolved, prefix + "/", service


@weight(1.5)
async def calc_code_bypassing_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Measure the amount of code that was pushed outside of pull requests."""
    try:
        filt = CodeFilter.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response

    async def login_loader() -> str:
        return (await request.user()).login

    repos, meta_ids = await resolve_repos(
        filt.in_, filt.account, request.uid, login_loader,
        request.sdb, request.mdb, request.cache, request.app["slack"])
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularity, filt.timezone)
    with_author = [s.rsplit("/", 1)[1] for s in (filt.with_author or [])]
    with_committer = [s.rsplit("/", 1)[1] for s in (filt.with_committer or [])]
    calculator = (await get_calculators_for_request(
        ["github"], filt.account, meta_ids, request))["github"]
    stats = await calculator.calc_code_metrics_github(
        FilterCommitsProperty.BYPASSING_PRS, time_intervals, repos, with_author,
        with_committer, filt.only_default_branch)  # type: List[CodeStats]
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


@weight(1.5)
async def calc_metrics_developers(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over developer activities."""
    try:
        filt = DeveloperMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    filters, all_repos = await _compile_filters_devs(
        filt.for_, request, filt.account, meta_ids)
    if filt.date_to < filt.date_from:
        raise ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        ))
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(all_repos)

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

    calculators = await get_calculators_for_request(
        {s for s, _ in filters}, filt.account, meta_ids, request)

    for service, (repos, devs, labels, jira, for_set) in filters:
        if for_set.aggregate_devgroups:
            dev_groups = [[devs[i] for i in group] for group in for_set.aggregate_devgroups]
        else:
            dev_groups = [[dev] for dev in devs]
        calculator = calculators[service]
        tasks.append(calculator.calc_developer_metrics_github(
            dev_groups, repos, time_intervals, topics, labels, jira, release_settings, prefixer))
        for_sets.append(for_set)
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


async def _compile_repos_releases(request: AthenianWebRequest,
                                  for_sets: List[List[str]],
                                  account: int,
                                  meta_ids: Tuple[int, ...],
                                  ) -> Tuple[List[Tuple[str, str, Tuple[Set[str], List[str]]]],
                                             Set[str]]:
    filters = []
    checkers = {}
    all_repos = set()
    async with request.sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos, prefix, service = await _extract_repos(
                request, account, meta_ids, for_set, i, all_repos, checkers, sdb_conn)
            filters.append((service, prefix, (set(chain.from_iterable(repos)), for_set)))
    return filters, all_repos


@weight(4)
async def calc_metrics_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate linear metrics over releases."""
    try:
        filt = ReleaseMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    filters, repos = await _compile_repos_releases(request, filt.for_, filt.account, meta_ids)
    grouped_for_sets = defaultdict(list)
    grouped_repos = defaultdict(list)
    for service, prefix, (for_set_repos, for_set) in filters:
        grouped_for_sets[service].append((prefix, for_set))
        grouped_repos[service].append(for_set_repos)
    del filters
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)

    tasks = [
        Settings.from_request(request, filt.account).list_release_matches(repos),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
        get_calculators_for_request(grouped_repos.keys(), filt.account, meta_ids, request),
        *(extract_release_participants(with_, meta_ids, request.mdb, position=i)
          for i, with_ in enumerate(filt.with_ or [])),
    ]
    release_settings, jira_ids, calculators, *participants = await gather(*tasks)
    met = []

    @sentry_span
    async def calculate_for_set_metrics(service, repos, for_sets):
        calculator = calculators[service]
        release_metric_values, release_matches = await calculator.calc_release_metrics_line_github(
            filt.metrics, time_intervals, filt.quantiles or (0, 1), repos, participants,
            LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude),
            JIRAFilter.from_web(filt.jira, jira_ids), release_settings, prefixer)
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


@weight(1)
async def calc_metrics_code_checks(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics on continuous integration runs, such as GitHub Actions, Jenkins, Circle, \
    etc."""
    try:
        filt = CodeCheckMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    meta_ids = await get_metadata_account_ids(filt.account, request.sdb, request.cache)
    filters = await compile_filters_checks(filt.for_, request, filt.account, meta_ids)
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
    calculators = await get_calculators_for_request(
        {s for s, _ in filters}, filt.account, meta_ids, request)

    @sentry_span
    async def calculate_for_set_metrics(service, repos, pusher_groups, labels, jira, for_set):
        calculator = calculators[service]
        metric_values, group_suite_counts, suite_sizes = \
            await calculator.calc_check_run_metrics_line_github(
                filt.metrics, time_intervals, filt.quantiles or (0, 1),
                repos, pusher_groups, filt.split_by_check_runs, labels, jira)
        mrange = range(len(met.metrics))
        for pushers_group_index, pushers_group in enumerate(metric_values):
            for repos_group_index, repos_group in enumerate(pushers_group):
                my_suite_counts = group_suite_counts[pushers_group_index, repos_group_index]
                total_group_suites = my_suite_counts.sum()
                for suite_size_group_index, suite_size_group in enumerate(repos_group):
                    group_for_set = for_set \
                        .select_pushers_group(pushers_group_index) \
                        .select_repogroup(repos_group_index)  # type: ForSetCodeChecks
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
                                confidence_scores=[mvs[i][m].confidence_score() for m in mrange],
                            ) for i, d in enumerate(ts[:-1])])
                        for v in cm.values:
                            if sum(1 for c in v.confidence_scores if c is not None) == 0:
                                v.confidence_mins = None
                                v.confidence_maxs = None
                                v.confidence_scores = None
                        met.calculated.append(cm)

    await gather(*(
        calculate_for_set_metrics(service, repos, ca_groups, labels, jira, for_set)
        for service, (repos, ca_groups, labels, jira, for_set) in filters
    ))
    return model_response(met)


@weight(2)
async def calc_metrics_deployments(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics on deployments submitted by `/events/deployments`."""
    raise NotImplementedError
