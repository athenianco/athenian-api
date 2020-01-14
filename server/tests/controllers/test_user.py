from datetime import datetime
import json

import dateutil.parser


async def test_get_user(client):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    items = json.loads(body)
    updated = items["updated"]
    del items["updated"]
    assert items == {
        "id": "auth0:vmarkovtsev",
        "email": "vadim@athenian.co",
        "name": "Vadim Markovtsev",
        "picture": ""
    }
    assert datetime.utcnow() >= dateutil.parser.parse(updated)
