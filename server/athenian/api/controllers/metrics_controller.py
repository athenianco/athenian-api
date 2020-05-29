import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from itertools import chain
from typing import List, Optional, Set, Tuple, Union

from aiohttp import web

from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.entries import METRIC_ENTRIES
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.miners.github.commit import FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import DeveloperTopic
from athenian.api.controllers.miners.pull_request_list_item import Participants, ParticipationKind
from athenian.api.controllers.reposet import resolve_repos, resolve_reposet
from athenian.api.controllers.settings import Settings
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.web import CalculatedDeveloperMetrics, CalculatedDeveloperMetricsItem, \
    CalculatedPullRequestMetrics, CalculatedPullRequestMetricsItem, \
    CalculatedPullRequestMetricValues, CodeBypassingPRsMeasurement, CodeFilter, \
    DeveloperMetricsRequest, ForSet, ForSetDevelopers, Granularity
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.models.web.pull_request_metrics_request import PullRequestMetricsRequest
# from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError

#           service                  developers
Filter = Tuple[str, Tuple[Set[str], Participants, ForSet]]
#                       repositories             originals


async def calc_metrics_pr_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate linear metrics over PRs.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    try:
        filt = PullRequestMetricsRequest.from_dict(body)  # type: PullRequestMetricsRequest
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    filters, repos = await _compile_repos_and_devs(filt.for_, request, filt.account)
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
    met.metrics = filt.metrics
    met.exclude_inactive = filt.exclude_inactive
    met.calculated = []
    # There should not be any new exception here so we don't have to catch ResponseError.
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    for service, (repos, devs, for_set) in filters:
        calcs = defaultdict(list)
        # for each filter, we find the functions to measure the metrics
        sentries = METRIC_ENTRIES[service]
        for m in filt.metrics:
            calcs[sentries[m]].append(m)
        gresults = []
        # for each metric, we find the function to calculate and call it
        for func, metrics in calcs.items():
            mvs = await func(metrics, time_intervals, repos, devs, filt.exclude_inactive,
                             release_settings, request.mdb, request.pdb, request.cache)
            assert len(mvs) == len(time_intervals)
            for mv, ts in zip(mvs, time_intervals):
                assert len(mv) == len(ts) - 1
                mr = {}
                gresults.append(mr)
                for i, m in enumerate(metrics):
                    mr[m] = [r[i] for r in mv]
        for granularity, results, ts in zip(filt.granularities, gresults, time_intervals):
            cm = CalculatedPullRequestMetricsItem(
                for_=for_set,
                granularity=granularity,
                values=[CalculatedPullRequestMetricValues(
                    date=(d - tzoffset).date(),
                    values=[results[m][i].value for m in met.metrics],
                    confidence_mins=[results[m][i].confidence_min for m in met.metrics],
                    confidence_maxs=[results[m][i].confidence_max for m in met.metrics],
                    confidence_scores=[results[m][i].confidence_score() for m in met.metrics],
                ) for i, d in enumerate(ts[:-1])])
            for v in cm.values:
                if sum(1 for c in v.confidence_scores if c is not None) == 0:
                    v.confidence_mins = None
                    v.confidence_maxs = None
                    v.confidence_scores = None
            met.calculated.append(cm)
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


async def _compile_repos_and_devs(for_sets: List[Union[ForSet, ForSetDevelopers]],
                                  request: AthenianWebRequest,
                                  account: int,
                                  ) -> (List[Filter], List[str]):
    """
    Build the list of Filter-s for a given list of ForSet-s.

    Repository sets are dereferenced. Access permissions are checked.

    :param for_sets: Paired lists of repositories and developers.
    :param request: Our incoming request to take the metadata DB, the user ID, the cache.
    :param account: Account ID on behalf of which we are loading reposets.
    :return: Resulting list of Filter-s and the list of all repositories after dereferencing, \
             with service prefixes.
    """
    filters = []
    sdb, user = request.sdb, request.uid
    checkers = {}
    all_repos = []
    async with sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos = set()
            service = None
            for repo in chain.from_iterable(await asyncio.gather(*[
                    resolve_reposet(r, ".for[%d].repositories[%d]" % (i, j), user, account, sdb,
                                    request.cache)
                    for j, r in enumerate(for_set.repositories)])):
                for key, prefix in PREFIXES.items():
                    if repo.startswith(prefix):
                        if service is None:
                            service = key
                        elif service != key:
                            raise ResponseError(InvalidRequestError(
                                detail='mixed providers are not allowed in the same "for" element',
                                pointer=".for[%d].repositories" % i,
                            ))
                        repos.add(repo[len(prefix):])
                        all_repos.append(repo)
            if service is None:
                raise ResponseError(InvalidRequestError(
                    detail='the provider of a "for" element is unsupported or the set is empty',
                    pointer=".for[%d].repositories" % i,
                ))
            checker = checkers.get(service)
            if checker is None:
                checker = await access_classes[service](
                    account, sdb_conn, request.mdb, request.cache).load()
                checkers[service] = checker
            denied = await checker.check(repos)
            if denied:
                raise ResponseError(InvalidRequestError(
                    detail="the following repositories are access denied for %s: %s" %
                           (service, denied),
                    pointer=".for[%d].repositories" % i,
                    status=HTTPStatus.FORBIDDEN,
                ))
            devs = {}
            prefix = PREFIXES[service]
            if isinstance(for_set, ForSet) and for_set.with_:
                for k, v in for_set.with_.items():
                    if not v:
                        continue
                    devs[ParticipationKind[k.upper()]] = dk = set()
                    for dev in v:
                        if not dev.startswith(prefix):
                            raise ResponseError(InvalidRequestError(
                                detail='providers in "with" and "repositories" do not match',
                                pointer=".for[%d].with" % i,
                            ))
                        dk.add(dev[len(prefix):])
            else:
                # DEPRECATED for /metrics/prs but not for /metrics/developers
                for dev in (for_set.developers or []):
                    if not dev.startswith(prefix):
                        raise ResponseError(InvalidRequestError(
                            detail='providers in "developers" and "repositories" do not match',
                            pointer=".for[%d].developers" % i,
                        ))
                    for pk in ParticipationKind:
                        devs.setdefault(pk, set()).add(dev[len(prefix):])
            filters.append((service, (repos, devs, for_set)))
    return filters, all_repos


async def calc_code_bypassing_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Measure the amount of code that was pushed outside of pull requests."""
    try:
        filt = CodeFilter.from_dict(body)  # type: CodeFilter
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
        filt = DeveloperMetricsRequest.from_dict(body)  # type: DeveloperMetricsRequest
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    # FIXME(vmarkovtsev): developer metrics + release settings???
    filters, _ = await _compile_repos_and_devs(filt.for_, request, filt.account)
    if filt.date_to < filt.date_from:
        return ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        )).response

    met = CalculatedDeveloperMetrics()
    met.date_from = filt.date_from
    met.date_to = filt.date_to
    met.timezone = filt.timezone
    met.metrics = filt.metrics
    met.calculated = []
    topics = {DeveloperTopic(t) for t in filt.metrics}
    time_from = datetime.combine(filt.date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(filt.date_to, datetime.max.time(), tzinfo=timezone.utc)
    if filt.timezone is not None:
        tzoffset = timedelta(minutes=-filt.timezone)
        time_from += tzoffset
        time_to += tzoffset
    for service, (repos, devs, for_set) in filters:
        devs = devs[ParticipationKind.AUTHOR]  # any key is fine, for example AUTHOR
        stats = await METRIC_ENTRIES[service]["developers"](
            devs, repos, topics, time_from, time_to, request.mdb, request.cache)
        met.calculated.append(CalculatedDeveloperMetricsItem(
            for_=for_set,
            values=[[getattr(s, DeveloperTopic(t).name) for t in filt.metrics] for s in stats],
        ))
    return model_response(met)
