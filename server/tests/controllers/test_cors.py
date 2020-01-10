async def test_cors(client):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "http://localhost",
    }
    response = await client.request(
        method="GET", path="/v1/reposet/1", headers=headers, json={},
    )
    assert response.headers["Access-Control-Expose-Headers"] == ""
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"

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
