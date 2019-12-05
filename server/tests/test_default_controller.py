# coding: utf-8

import pytest
import json
from aiohttp import web

from athenian.api.models.calculated_metrics import CalculatedMetrics
from athenian.api.models.invalid_request_error import InvalidRequestError
from athenian.api.models.metrics_request import MetricsRequest
from athenian.api.models.no_source_data_error import NoSourceDataError


async def test_calc_metrics(client):
    """Test case for calc_metrics

    Calculate metrics.
    """
    body = {
  "for" : [ {
    "developers" : [ "github.com/vmarkovtsev", "github.com/mcuadros" ],
    "repositories" : [ "github.com/src-d/hercules", "github.com/athenianco/athenian-api" ]
  }, {
    "developers" : [ "github.com/vmarkovtsev", "github.com/mcuadros" ],
    "repositories" : [ "github.com/src-d/hercules", "github.com/athenianco/athenian-api" ]
  } ],
  "metrics" : [ null, null ],
  "date_to" : "2000-01-23",
  "date_from" : "2000-01-23"
}
    headers = { 
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    response = await client.request(
        method='POST',
        path='/v1/metrics',
        headers=headers,
        json=body,
        )
    assert response.status == 200, 'Response body is : ' + (await response.read()).decode('utf-8')

