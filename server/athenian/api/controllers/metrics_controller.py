import asyncio
from collections import defaultdict
from http import HTTPStatus
from itertools import chain
from typing import List, Set, Tuple

from aiohttp import web

from athenian.api.controllers.features.entries import METRIC_ENTRIES
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.reposet import resolve_reposet
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.web import CalculatedMetric, CalculatedMetrics, CalculatedMetricValues, \
    ForSet, Granularity
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.models.web.pull_request_metrics_request import PullRequestMetricsRequest
# from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import response, ResponseError

#           service                  developers
Filter = Tuple[str, Tuple[Set[str], List[str], ForSet]]
#                       repositories          originals


async def calc_metrics_pr_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate linear metrics over PRs.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    try:
        body = PullRequestMetricsRequest.from_dict(body)
    except ValueError as e:
        return ResponseError(InvalidRequestError("?", detail=str(e))).response

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
    met = CalculatedMetrics()
    met.date_from = body.date_from
    met.date_to = body.date_to
    met.granularity = body.granularity
    met.metrics = body.metrics
    met.calculated = []

    try:
        filters = await _compile_filters(body.for_, request, body.account)
    except ResponseError as e:
        return e.response
    if body.date_to < body.date_from:
        return ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        )).response
    try:
        time_intervals = Granularity.split(body.granularity, body.date_from, body.date_to)
    except ValueError:
        return ResponseError(InvalidRequestError(
            detail="granularity value does not match /%s/" % Granularity.format.pattern,
            pointer=".granularity",
        )).response
    for service, (repos, devs, for_set) in filters:
        calcs = defaultdict(list)
        # for each filter, we find the functions to measure the metrics
        sentries = METRIC_ENTRIES[service]
        for m in body.metrics:
            calcs[sentries[m]].append(m)
        results = {}
        # for each metric, we find the function to calculate and call it
        for func, metrics in calcs.items():
            fres = await func(metrics, time_intervals, repos, devs, request.mdb, request.cache)
            assert len(fres) == len(time_intervals) - 1
            for i, m in enumerate(metrics):
                results[m] = [r[i] for r in fres]
        cm = CalculatedMetric(
            for_=for_set,
            values=[CalculatedMetricValues(
                date=d,
                values=[results[m][i].value for m in met.metrics],
                confidence_mins=[results[m][i].confidence_min for m in met.metrics],
                confidence_maxs=[results[m][i].confidence_max for m in met.metrics],
                confidence_scores=[results[m][i].confidence_score() for m in met.metrics],
            ) for i, d in enumerate(time_intervals[1:])])
        for v in cm.values:
            if sum(1 for c in v.confidence_scores if c is not None) == 0:
                v.confidence_mins = None
                v.confidence_maxs = None
                v.confidence_scores = None
        met.calculated.append(cm)
    return response(met)


async def _compile_filters(for_sets: List[ForSet], request: AthenianWebRequest, account: int,
                           ) -> List[Filter]:
    filters = []
    sdb, user = request.sdb, request.uid
    checkers = {}
    async with sdb.connection() as sdb_conn:
        for i, for_set in enumerate(for_sets):
            repos = set()
            devs = []
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
            for dev in (for_set.developers or []):
                for key, prefix in PREFIXES.items():
                    if dev.startswith(prefix):
                        if service != key:
                            raise ResponseError(InvalidRequestError(
                                detail='mixed providers are not allowed in the same "for" element',
                                pointer=".for[%d].developers" % i,
                            ))
                        devs.append(dev[len(prefix):])
            filters.append((service, (repos, devs, for_set)))
    return filters


async def calc_code_bypassing_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Measure the amount of code that was pushed outside of pull requests."""
    return None
