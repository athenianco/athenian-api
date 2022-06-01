class TestGraphiQL:
    async def test(self, client) -> None:
        response = await client.request(method="GET", path="/align/ui")
        html = (await response.read()).decode("utf-8")

        assert "<html>" in html
        assert "graphiql.min.js" in html
