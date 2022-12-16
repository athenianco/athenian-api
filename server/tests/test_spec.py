async def test_spec_load(client, headers):
    response = await client.request(method="GET", path="/v1/openapi.json", headers=headers)
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    assert "openapi" in body
    for tmpl in ("server_description", "server_version", "server_url"):
        assert ("{{ %s }}" % tmpl) not in body
