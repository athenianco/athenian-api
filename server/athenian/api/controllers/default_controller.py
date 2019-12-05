from typing import List, Dict
from aiohttp import web

from athenian.api.models.calculated_metrics import CalculatedMetrics
from athenian.api.models.invalid_request_error import InvalidRequestError
from athenian.api.models.metrics_request import MetricsRequest
from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api import util


async def calc_metrics(request: web.Request, body) -> web.Response:
    """Calculate metrics.

    

    :param body: Desired metric definitions.
    :type body: dict | bytes

    """
    body = MetricsRequest.from_dict(body)
    return web.Response(status=200)
