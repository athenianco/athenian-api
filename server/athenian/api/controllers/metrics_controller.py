import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from itertools import chain
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from aiohttp import web
import databases.core

from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.entries import METRIC_ENTRIES
from athenian.api.controllers.jira_controller import get_jira_installation
from athenian.api.controllers.miners.access_classes import access_classes, AccessChecker
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.commit import FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import DeveloperTopic
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind, \
    ReleaseParticipationKind
from athenian.api.controllers.reposet import resolve_repos, resolve_reposet
from athenian.api.controllers.settings import Settings
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.web import CalculatedDeveloperMetrics, CalculatedDeveloperMetricsItem, \
    CalculatedLinearMetricValues, CalculatedPullRequestMetrics, CalculatedPullRequestMetricsItem, \
    CalculatedReleaseMetric, CodeBypassingPRsMeasurement, CodeFilter, DeveloperMetricsRequest, \
    ForSet, ForSetDevelopers, Granularity, ReleaseMetricsRequest
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.models.web.pull_request_metrics_request import PullRequestMetricsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span

#               service                       developers                           originals
FilterPRs = Tuple[str, Tuple[List[Set[str]], PRParticipants, LabelFilter, JIRAFilter, ForSet]]
#                             repositories

#                service                     developers
FilterDevs = Tuple[str, Tuple[List[Set[str]], List[str], ForSetDevelopers]]
#                              repositories                  originals


async def calc_metrics_pr_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate linear metrics over PRs.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    try:
        filt = PullRequestMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    filters, repos = await compile_repos_and_devs_prs(filt.for_, request, filt.account)
    time_intervals, tzoffset = _split_to_time_intervals(
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

    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)

    @sentry_span
    async def calculate_for_set_metrics(service, repos, devs, labels, jira, for_set):
        ti_mvs = await METRIC_ENTRIES[service]["prs_linear"](
            filt.metrics, time_intervals, filt.quantiles or (0, 1),
            repos, devs, labels, jira,
            filt.exclude_inactive, release_settings, filt.fresh,
            request.mdb, request.pdb, request.cache)
        assert len(ti_mvs) == len(time_intervals)
        mrange = range(len(met.metrics))
        for granularity, ts, group_mvs in zip(filt.granularities, time_intervals, ti_mvs):
            assert len(group_mvs) == len(repos)
            for group, mvs in enumerate(group_mvs):
                cm = CalculatedPullRequestMetricsItem(
                    for_=for_set.select_repogroup(group),
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

    tasks = []
    for service, (repos, devs, labels, jira, for_set) in filters:
        tasks.append(calculate_for_set_metrics(service, repos, devs, labels, jira, for_set))
    if len(tasks) == 1:
        await tasks[0]
    else:
        for err in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(err, Exception):
                raise err from None
    return model_response(met)


def _split_to_time_intervals(date_from: date,
                             date_to: date,
                             granularities: Union[str, List[str]],
                             tzoffset: Optional[int],
                             ) -> Tuple[Union[List[datetime], List[List[datetime]]], timedelta]:
    if date_to < date_from:
        raise ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        ))
    tzoffset = timedelta(minutes=-tzoffset) if tzoffset is not None else timedelta(0)

    def split(granularity: str, ptr: str) -> List[datetime]:
        try:
            intervals = Granularity.split(granularity, date_from, date_to)
        except ValueError:
            raise ResponseError(InvalidRequestError(
                detail='granularity "%s" does not match /%s/' % (
                    granularity, Granularity.format.pattern),
                pointer=ptr,
            ))
        return [datetime.combine(i, datetime.min.time(), tzinfo=timezone.utc) + tzoffset
                for i in intervals]

    if isinstance(granularities, str):
        return split(granularities, ".granularity"), tzoffset

    return [split(g, ".granularities[%d]" % i) for i, g in enumerate(granularities)], tzoffset


async def compile_repos_and_devs_prs(for_sets: List[ForSet],
                                     request: AthenianWebRequest,
                                     account: int,
                                     ) -> Tuple[List[FilterPRs], Set[str]]:
    """
    Build the list of filters for a given list of ForSet-s.

    Repository sets are dereferenced. Access permissions are checked.

    :param for_sets: Paired lists of repositories, developers, and PR labels.
    :param request: Our incoming request to take the metadata DB, the user ID, the cache.
    :param account: Account ID on behalf of which we are loading reposets.
    :return: Resulting list of filters and the set of all repositories after dereferencing, \
             with service prefixes.
    """
    filters = []
    checkers = {}
    all_repos = set()
    async with request.sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos, service = await _extract_repos(
                request, account, for_set.repositories, i, all_repos, checkers, sdb_conn)
            if for_set.repogroups is not None:
                repogroups = [set(chain.from_iterable(repos[i] for i in group))
                              for group in for_set.repogroups]
            else:
                repogroups = [set(chain.from_iterable(repos))]
            prefix = PREFIXES[service]
            devs = {}
            for k, v in (for_set.with_ or {}).items():
                if not v:
                    continue
                devs[PRParticipationKind[k.upper()]] = dk = set()
                for dev in v:
                    if not dev.startswith(prefix):
                        raise ResponseError(InvalidRequestError(
                            detail='providers in "with" and "repositories" do not match',
                            pointer=".for[%d].with" % i,
                        ))
                    dk.add(dev[len(prefix):])
            labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
            try:
                jira = JIRAFilter.from_web(
                    for_set.jira, await get_jira_installation(account, request.sdb, request.cache))
            except ResponseError:
                jira = JIRAFilter.empty()
            filters.append((service, (repogroups, devs, labels, jira, for_set)))
    return filters, all_repos


async def _compile_repos_and_devs_devs(for_sets: List[ForSetDevelopers],
                                       request: AthenianWebRequest,
                                       account: int,
                                       ) -> (List[FilterDevs], List[str]):
    """
    Build the list of filters for a given list of ForSetDevelopers'.

    Repository sets are de-referenced. Access permissions are checked.

    :param for_sets: Paired lists of repositories and developers.
    :param request: Our incoming request to take the metadata DB, the user ID, the cache.
    :param account: Account ID on behalf of which we are loading reposets.
    :return: Resulting list of filters and the list of all repositories after dereferencing, \
             with service prefixes.
    """
    filters = []
    checkers = {}
    all_repos = set()
    async with request.sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos, service = await _extract_repos(
                request, account, for_set.repositories, i, all_repos, checkers, sdb_conn)
            if for_set.repogroups is not None:
                repogroups = [set(chain.from_iterable(repos[i] for i in group))
                              for group in for_set.repogroups]
            else:
                repogroups = [set(chain.from_iterable(repos))]
            prefix = PREFIXES[service]
            devs = []
            for dev in for_set.developers:
                if not dev.startswith(prefix):
                    raise ResponseError(InvalidRequestError(
                        detail='providers in "developers" and "repositories" do not match',
                        pointer=".for[%d].developers" % i,
                    ))
                devs.append(dev[len(prefix):])
            labels = LabelFilter.from_iterables(for_set.labels_include, for_set.labels_exclude)
            try:
                jira = JIRAFilter.from_web(
                    for_set.jira, await get_jira_installation(account, request.sdb, request.cache))
            except ResponseError:
                jira = JIRAFilter.empty()
            filters.append((service, (repogroups, devs, labels, jira, for_set)))
    return filters, all_repos


async def _extract_repos(request: AthenianWebRequest,
                         account: int,
                         for_set: List[str],
                         for_set_index: int,
                         all_repos: Set[str],
                         checkers: Dict[str, AccessChecker],
                         sdb: databases.core.Connection) -> Tuple[Sequence[Set[str]], str]:
    user = request.uid
    service = None
    resolved = await asyncio.gather(*[
        resolve_reposet(r, ".for[%d].repositories[%d]" % (
            for_set_index, j), user, account, sdb, request.cache)
        for j, r in enumerate(for_set)
    ], return_exceptions=True)
    for repos in resolved:
        if isinstance(repos, Exception):
            raise repos from None
        for i, repo in enumerate(repos):
            for key, prefix in PREFIXES.items():
                if repo.startswith(prefix):
                    if service is None:
                        service = key
                    elif service != key:
                        raise ResponseError(InvalidRequestError(
                            detail='mixed providers are not allowed in the same "for" element',
                            pointer=".for[%d].repositories" % for_set_index,
                        ))
                    repos[i] = repo[len(prefix):]
                    all_repos.add(repo)
    if service is None:
        raise ResponseError(InvalidRequestError(
            detail='the provider of a "for" element is unsupported or the set is empty',
            pointer=".for[%d].repositories" % for_set_index,
        ))
    if (checker := checkers.get(service)) is None:
        checker = await access_classes[service](account, sdb, request.mdb, request.cache).load()
        checkers[service] = checker
    if denied := await checker.check(set(chain.from_iterable(resolved))):
        raise ResponseError(InvalidRequestError(
            detail="the following repositories are access denied for %s: %s" % (service, denied),
            pointer=".for[%d].repositories" % for_set_index,
            status=HTTPStatus.FORBIDDEN,
        ))
    return resolved, service


async def calc_code_bypassing_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Measure the amount of code that was pushed outside of pull requests."""
    try:
        filt = CodeFilter.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    repos = await resolve_repos(
        filt.in_, filt.account, request.uid, request.native_uid,
        request.sdb, request.mdb, request.cache, request.app["slack"])
    time_intervals, tzoffset = _split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularity, filt.timezone)
    with_author = [s.split("/", 1)[1] for s in (filt.with_author or [])]
    with_committer = [s.split("/", 1)[1] for s in (filt.with_committer or [])]
    stats = await METRIC_ENTRIES["github"]["code"](
        FilterCommitsProperty.BYPASSING_PRS, time_intervals, repos, with_author, with_committer,
        request.mdb, request.cache)  # type: List[CodeStats]
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


async def calc_metrics_developer(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over developer activities."""
    try:
        filt = DeveloperMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    filters, all_repos = await _compile_repos_and_devs_devs(filt.for_, request, filt.account)
    if filt.date_to < filt.date_from:
        raise ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        ))
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(all_repos)

    met = CalculatedDeveloperMetrics()
    met.date_from = filt.date_from
    met.date_to = filt.date_to
    met.timezone = filt.timezone
    met.metrics = filt.metrics
    met.calculated = []
    topics = {DeveloperTopic(t) for t in filt.metrics}
    time_from, time_to = filt.resolve_time_from_and_to()
    tasks = []
    for_sets = []
    for service, (repos, devs, labels, jira, for_set) in filters:
        tasks.append(METRIC_ENTRIES[service]["developers"](
            devs, repos, time_from, time_to, topics, labels, jira, release_settings,
            request.mdb, request.pdb, request.cache))
        for_sets.append(for_set)
    all_stats = await asyncio.gather(*tasks, return_exceptions=True)
    for stats, for_set in zip(all_stats, for_sets):
        if isinstance(stats, Exception):
            raise stats from None
        for i, group in enumerate(stats):
            met.calculated.append(CalculatedDeveloperMetricsItem(
                for_=for_set.select_repogroup(i),
                values=[[getattr(s, DeveloperTopic(t).name) for t in filt.metrics] for s in group],
            ))
    return model_response(met)


async def _compile_repos_releases(request: AthenianWebRequest,
                                  for_sets: List[List[str]],
                                  account: int,
                                  ) -> Tuple[List[Tuple[str, Tuple[Set[str], List[str]]]],
                                             Set[str]]:
    filters = []
    checkers = {}
    all_repos = set()
    async with request.sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos, service = await _extract_repos(
                request, account, for_set, i, all_repos, checkers, sdb_conn)
            filters.append((service, (set(chain.from_iterable(repos)), for_set)))
    return filters, all_repos


async def calc_metrics_releases_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate linear metrics over releases."""
    try:
        filt = ReleaseMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    filters, all_repos = await _compile_repos_releases(request, filt.for_, filt.account)
    grouped_for_sets = defaultdict(list)
    grouped_repos = defaultdict(list)
    for service, (repos, for_set) in filters:
        grouped_for_sets[service].append(for_set)
        grouped_repos[service].append(repos)
    del filters
    time_intervals, tzoffset = _split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(all_repos)
    met = []

    @sentry_span
    async def calculate_for_set_metrics(service, repos, for_sets):
        participants = {
            rpk: getattr(filt.with_, attr) or []
            for attr, rpk in (("releaser", ReleaseParticipationKind.RELEASER),
                              ("pr_author", ReleaseParticipationKind.PR_AUTHOR),
                              ("commit_author", ReleaseParticipationKind.COMMIT_AUTHOR))
        } if filt.with_ is not None else {}
        ti_mvs, release_matches = await METRIC_ENTRIES[service]["releases_linear"](
            filt.metrics, time_intervals, filt.quantiles or (0, 1), repos, participants,
            release_settings, request.mdb, request.pdb, request.cache)
        release_matches = {k: v.name for k, v in release_matches.items()}
        mrange = range(len(filt.metrics))
        assert len(ti_mvs) == len(time_intervals)
        for granularity, ts, group_mvs in zip(filt.granularities, time_intervals, ti_mvs):
            assert len(group_mvs) == len(for_sets)
            for mvs, for_set, my_repos in zip(group_mvs, for_sets, repos):
                my_release_matches = {}
                for r in my_repos:
                    r = PREFIXES[service] + r
                    try:
                        my_release_matches[r] = release_matches[r]
                    except KeyError:
                        continue
                cm = CalculatedReleaseMetric(
                    for_=for_set,
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

    tasks = []
    for service, repos in grouped_repos.items():
        tasks.append(calculate_for_set_metrics(service, repos, grouped_for_sets[service]))
    if len(tasks) == 1:
        await tasks[0]
    else:
        for err in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(err, Exception):
                raise err from None
    return model_response(met)
