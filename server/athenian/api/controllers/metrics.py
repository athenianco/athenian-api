from collections import defaultdict

from aiohttp import web

from athenian.api import FriendlyJson
from athenian.api.controllers.features.entries import ENTRIES
from athenian.api.models import CalculatedMetric, CalculatedMetricValues, ForSet, RepositorySet
from athenian.api.models.calculated_metrics import CalculatedMetrics
from athenian.api.models.metadata import PREFIXES
# from athenian.api.models.invalid_request_error import InvalidRequestError
from athenian.api.models.metrics_request import MetricsRequest
# from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api.typing_utils import AthenianWebRequest


async def calc_metrics_line(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    body = MetricsRequest.from_dict(body)

    met = CalculatedMetrics()
    met.date_from = body.date_from
    met.date_to = body.date_to
    met.granularity = body.granularity
    met.metrics = body.metrics
    met.calculated = []

    repos = defaultdict(list)
    for for_set in body._for:
        for repo in for_set.repositories:
            for service, prefix in PREFIXES.items():
                if repo.startswith(prefix):
                    repos[service].append(repo[repo[len(prefix):]])
    time_intervals = body.granularity.split(body.date_from, body.date_to)
    for service, srepos in repos.items():
        calcs = defaultdict(list)
        sentries = ENTRIES[service]
        for m in body.metrics:
            calcs[sentries[m.to_str()]].append(m.to_str())
        results = {}
        for func, metrics in calcs.items():
            fres = func(metrics, srepos, time_intervals, request.mdb)
            for i, m in enumerate(metrics):
                results[m] = [r[i] for r in fres]
        met.calculated.append(CalculatedMetric(
            _for=ForSet(repositories=RepositorySet.from_dict(srepos)),
            values=[CalculatedMetricValues(
                date=d,
                values=[results[m][i].value for m in met.metrics],
                confidence_mins=[results[m][i].confidence_min for m in met.metrics],
                confidence_maxs=[results[m][i].confidence_max for m in met.metrics],
                confidence_scores=[results[m][i].confidence_score() for m in met.metrics],
            ) for i, d in enumerate(time_intervals[1:])]))
    return web.json_response(met.to_dict(), dumps=FriendlyJson.dumps)
