import asyncio
from time import time

import pytest

from athenian.api.async_utils import gather


# TODO: fix response validation against the schema
@pytest.mark.flaky(reruns=5, reruns_delay=1)
@pytest.mark.app_validate_responses(False)
async def test_heavy_load(client, headers):
    req_body = {
        "for": [
            {
                "with": {
                    "author": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                },
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": ["pr-lead-time"],
        "date_from": "2018-10-13",
        "date_to": "2020-01-23",
        "granularities": ["week"],
        "exclude_inactive": False,
        "account": 1,
    }

    async def request_200():
        return await client.request(
            method="POST", path="/v1/metrics/pull_requests", headers=headers, json=req_body,
        )

    async def request_503():
        await asyncio.sleep(0.1)
        return await request_200()

    response1, response2 = await gather(request_200(), request_503())
    assert response1.status == 200
    assert response2.status == 503

    response3 = await request_200()
    assert response3.status == 200


@pytest.mark.app_validate_responses(False)
@pytest.mark.flaky(reruns=5, reruns_delay=1)
async def test_timeout(client, headers, app):
    req_body = {
        "for": [
            {
                "with": {
                    "author": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                },
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": ["pr-lead-time"],
        "date_from": "2018-10-13",
        "date_to": "2020-01-23",
        "granularities": ["week"],
        "exclude_inactive": False,
        "account": 1,
    }
    app.TIMEOUT = 0

    start_time = time()
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=req_body,
    )
    assert (time() - start_time) < 0.5
    assert response.status == 500
    assert (await response.json())["type"] == "/errors/ServerTimeout"
