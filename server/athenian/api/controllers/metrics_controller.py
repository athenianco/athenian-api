from collections import defaultdict
from typing import List, Tuple

from aiohttp import web

from athenian.api.controllers.features.entries import ENTRIES
from athenian.api.controllers.response import response, ResponseError
from athenian.api.models import CalculatedMetric, CalculatedMetricValues, ForSet, Granularity
from athenian.api.models.calculated_metrics import CalculatedMetrics
from athenian.api.models.invalid_request_error import InvalidRequestError
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metrics_request import MetricsRequest
# from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api.typing_utils import AthenianWebRequest


#           service                  developers
Filter = Tuple[str, Tuple[List[str], List[str]]]
#                        repositories


async def calc_metrics_line(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    body = MetricsRequest.from_dict(body)

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
        filters = _compile_filters(body._for)
    except ResponseError as e:
        return e.response
    if body.date_to < body.date_from:
        err = InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        )
        raise ResponseError(err, 400)
    time_intervals = Granularity.split(body.granularity, body.date_from, body.date_to)
    for service, (repos, devs) in filters:
        calcs = defaultdict(list)
        # for each filter, we find the functions to measure the metrics
        sentries = ENTRIES[service]
        for m in body.metrics:
            calcs[sentries[m]].append(m)
        results = {}
        # for each metric, we find the function to calculate and call it
        for func, metrics in calcs.items():
            fres = await func(metrics, time_intervals, repos, devs, request.mdb)
            for i, m in enumerate(metrics):
                results[m] = [r[i] for r in fres]
        met.calculated.append(CalculatedMetric(
            _for=ForSet(repositories=repos, developers=devs),
            values=[CalculatedMetricValues(
                date=d,
                values=[results[m][i].value for m in met.metrics],
                confidence_mins=[results[m][i].confidence_min for m in met.metrics],
                confidence_maxs=[results[m][i].confidence_max for m in met.metrics],
                confidence_scores=[results[m][i].confidence_score() for m in met.metrics],
            ) for i, d in enumerate(time_intervals[1:])]))
    return response(met)


def _compile_filters(for_sets) -> List[Filter]:
    filters = []
    for i, for_set in enumerate(for_sets):
        repos = []
        devs = []
        service = None
        for repo in for_set.repositories:
            for key, prefix in PREFIXES.items():
                if repo.startswith(prefix):
                    if service is None:
                        service = key
                    elif service != key:
                        err = InvalidRequestError(
                            detail='mixed providers are not allowed in the same "for" element',
                            pointer=".for[%d].repositories" % i,
                        )
                        raise ResponseError(err, 400)
                    repos.append(repo[len(prefix):])
        if service is None:
            err = InvalidRequestError(
                detail='the provider of a "for" element is unsupported or the set is empty',
                pointer=".for[%d].repositories" % i,
            )
            raise ResponseError(err, 400)
        for dev in for_set.developers:
            for key, prefix in PREFIXES.items():
                if dev.startswith(prefix):
                    if service != key:
                        err = InvalidRequestError(
                            detail='mixed providers are not allowed in the same "for" element',
                            pointer=".for[%d].developers" % i,
                        )
                        raise ResponseError(err, 400)
                    devs.append(dev[len(prefix):])
        filters.append((service, (repos, devs)))
    return filters
