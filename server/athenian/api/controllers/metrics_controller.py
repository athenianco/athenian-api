from collections import defaultdict
from itertools import chain
from typing import List, Tuple

from aiohttp import web

from athenian.api.controllers.features.entries import ENTRIES
from athenian.api.controllers.reposet import resolve_reposet
from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.web import CalculatedMetric, CalculatedMetrics, CalculatedMetricValues, \
    ForSet, Granularity
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.models.web.metrics_request import MetricsRequest
# from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api.request import AthenianWebRequest

#           service                  developers
Filter = Tuple[str, Tuple[List[str], List[str], ForSet]]
#                        repositories          originals


async def calc_metrics_line(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    try:
        body = MetricsRequest.from_dict(body)
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
        filters = await _compile_filters(body._for, request, body.account)
    except ResponseError as e:
        return e.response
    if body.date_to < body.date_from:
        raise ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        ))
    time_intervals = Granularity.split(body.granularity, body.date_from, body.date_to)
    for service, (repos, devs, for_set) in filters:
        calcs = defaultdict(list)
        # for each filter, we find the functions to measure the metrics
        sentries = ENTRIES[service]
        for m in body.metrics:
            calcs[sentries[m]].append(m)
        results = {}
        # for each metric, we find the function to calculate and call it
        for func, metrics in calcs.items():
            fres = await func(metrics, time_intervals, repos, devs, request.mdb)
            assert len(fres) == len(time_intervals) - 1
            for i, m in enumerate(metrics):
                results[m] = [r[i] for r in fres]
        met.calculated.append(CalculatedMetric(
            _for=for_set,
            values=[CalculatedMetricValues(
                date=d,
                values=[results[m][i].value for m in met.metrics],
                confidence_mins=[results[m][i].confidence_min for m in met.metrics],
                confidence_maxs=[results[m][i].confidence_max for m in met.metrics],
                confidence_scores=[results[m][i].confidence_score() for m in met.metrics],
            ) for i, d in enumerate(time_intervals[1:])]))
    return response(met)


async def _compile_filters(for_sets: List[ForSet], request: AthenianWebRequest, account: int,
                           ) -> List[Filter]:
    filters = []
    sdb, user = request.sdb, request.uid()
    for i, for_set in enumerate(for_sets):
        repos = []
        devs = []
        service = None
        for repo in chain.from_iterable([
                await resolve_reposet(r, ".for[%d].repositories" % i, sdb, user, account)
                for r in for_set.repositories]):
            for key, prefix in PREFIXES.items():
                if repo.startswith(prefix):
                    if service is None:
                        service = key
                    elif service != key:
                        raise ResponseError(InvalidRequestError(
                            detail='mixed providers are not allowed in the same "for" element',
                            pointer=".for[%d].repositories" % i,
                        ))
                    repos.append(repo[len(prefix):])
        if service is None:
            raise ResponseError(InvalidRequestError(
                detail='the provider of a "for" element is unsupported or the set is empty',
                pointer=".for[%d].repositories" % i,
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
