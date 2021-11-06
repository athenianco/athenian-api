import asyncio

from athenian.api.async_utils import gather


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
            method="POST", path="/v1/metrics/pull_requests", headers=headers, json=req_body)

    async def request_503():
        await asyncio.sleep(0.1)
        return await request_200()

    response1, response2 = await gather(request_200(), request_503())
    assert response1.status == 200
    assert response2.status == 503

    response3 = await request_200()
    assert response3.status == 200
