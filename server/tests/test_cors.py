async def test_cors_headers(client, headers):
    response = await client.request(
        method="GET", path="/v1/reposet/1", headers=headers, json={},
    )
    assert set(response.headers["Access-Control-Expose-Headers"].split(",")) == \
        {"X-Performance-DB", "X-Backend-Server", "X-Performance-Precomputed-Hits",
         "X-Performance-Precomputed-Misses",
         "Content-Length", "Server", "Date", "User"}
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"
    assert response.headers["X-Backend-Server"]
    assert response.headers["X-Performance-DB"]

    # preflight
    headers = {
        "Access-Control-Request-Headers": "Content-Type",
        "Access-Control-Request-Method": "GET",
        "Origin": "http://localhost",
    }
    response = await client.request(
        method="OPTIONS", path="/v1/reposet/1", headers=headers, json={},
    )
    assert response.headers["Access-Control-Allow-Headers"] == "CONTENT-TYPE"
    assert response.headers["Access-Control-Allow-Methods"] == "GET"
    assert response.headers["Access-Control-Max-Age"] == "3600"
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"


async def test_cors_cache(client, headers, client_cache):
    response = await client.request(
        method="GET", path="/v1/reposet/1", headers=headers, json={},
    )
    assert set(response.headers["Access-Control-Expose-Headers"].split(",")) == \
        {"X-Performance-DB", "X-Backend-Server", "X-Performance-Cache-Ignores",
         "X-Performance-Cache-Hits", "X-Performance-Cache-Misses",
         "X-Performance-Precomputed-Hits", "X-Performance-Precomputed-Misses",
         "Content-Length", "Server", "Date", "User"}
