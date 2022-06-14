import athenian.api
import athenian.api.__main__


async def test_manhole_smoke(client, headers, app, client_cache):
    assert athenian.api.__main__.trace_sample_rate_manhole(None) is None
    code = """from sqlalchemy import select, func
from athenian.api.models.state.models import Account
import athenian.api

athenian.api.__main__.trace_sample_rate_manhole = lambda request: 0
accounts = await request.sdb.fetch_val(select([func.count(Account.id)]))
response = await handler(request)
response.headers.add("X-Accounts", str(accounts))"""
    await client_cache.set(b"manhole", code.encode())
    response = await client.request(method="GET", path="/v1/versions", headers=headers, json={})
    assert response.headers["X-Accounts"] == "4"
    assert athenian.api.__main__.trace_sample_rate_manhole(None) == 0


async def test_manhole_error(client, headers, app, client_cache):
    code = """raise AssertionError("fail")"""
    await client_cache.set(b"manhole", code.encode())
    response = await client.request(method="GET", path="/v1/versions", headers=headers, json={})
    assert response.status == 200


async def test_manhole_response_error(client, headers, app, client_cache):
    code = """from athenian.api.models.web import NotFoundError
raise ResponseError(NotFoundError(detail="test"))"""
    await client_cache.set(b"manhole", code.encode())
    response = await client.request(method="GET", path="/v1/versions", headers=headers, json={})
    assert response.status == 404
