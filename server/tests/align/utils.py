from aiohttp.test_utils import TestClient


async def align_graphql_request(client: TestClient, **kwargs) -> dict:
    """Execute a request to the align graphql endpoint."""
    response = await client.request(method="POST", path="/align/graphql", **kwargs)
    assert response.status == 200
    return await response.json()


def get_extension_error(response: dict) -> str:
    """Return the first extension error of the graphql response."""
    return response["errors"][0]["extensions"]["detail"]


def assert_extension_error(response: dict, error: str) -> None:
    """Check the first extension error of the graphql response."""
    assert get_extension_error(response) == error
