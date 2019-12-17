import datetime

from aiohttp import web
from sqlalchemy import select, sql

from athenian.api import FriendlyJson
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.models import CalculatedMetric, CalculatedMetricValues
from athenian.api.models.calculated_metrics import CalculatedMetrics
from athenian.api.models.metadata import github
# from athenian.api.models.invalid_request_error import InvalidRequestError
from athenian.api.models.metrics_request import MetricsRequest
# from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api.typing_utils import AthenianWebRequest


async def calc_metrics(request: AthenianWebRequest, body) -> web.Response:
    """Calculate metrics.

    :param request: HTTP request.
    :param body: Desired metric definitions.
    :type body: dict | bytes
    """
    # The following is a technology demo that we can work with a DB, parse the request and respond
    # with something sensible.
    body = MetricsRequest.from_dict(body)
    github_repos_query = []
    for for_set in body._for:
        for repo in for_set.repositories:
            if repo.startswith("github.com/"):
                github_repos_query.append(github.Repository.full_name == repo[len("github.com/"):])
    github_repos_query = select([github.Repository]).where(sql.or_(*github_repos_query))
    repos = await read_sql_query(github_repos_query, request.mdb)
    print(repos)  # demo output, to be removed with the rest of the code
    met = CalculatedMetrics()
    met.date_from = body.date_from
    met.date_to = body.date_to
    met.granularity = body.granularity
    met.metrics = body.metrics
    met.calculated = []
    now = datetime.date.today()
    for for_set in body._for:
        met.calculated.append(CalculatedMetric(
            _for=for_set,
            values=[CalculatedMetricValues(date=now, values=[0.5] * len(met.metrics),
                                           confidence_scores=[75] * len(met.metrics))] * 10))
    return web.json_response(met.to_dict(), dumps=FriendlyJson.dumps)
