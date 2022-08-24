from typing import Sequence

from aiohttp.test_utils import TestClient


async def align_graphql_request(client: TestClient, **kwargs) -> dict:
    """Execute a request to the align graphql endpoint."""
    response = await client.request(method="POST", path="/align/graphql", **kwargs)
    assert response.status == 200
    return await response.json()


def get_extension_error_obj(response: dict) -> dict:
    """Return the first extension error object of the graphql response."""
    return response["errors"][0]["extensions"]


def assert_extension_error(response: dict, error: str) -> None:
    """Check the first extension error of the graphql response."""
    assert error in get_extension_error_obj(response)["detail"]


def build_recursive_fields_structure(
    fields: Sequence[str],
    depth: int,
    recursive_field: str = "children",
) -> str:
    result = "\n".join(fields)
    for i in range(depth - 1):
        indent = " " * 4 * i
        result = f"""
            {result}
            {indent}{recursive_field} {{
        """.strip()
        for field in fields:
            result += f"{indent}    {field}\n"

    for i in range(depth - 1):
        indent = " " * 4 * (depth - 1 - i)
        result = f"""
            {result}
            {indent}}}
        """.strip()

    return result


def build_fragment(name: str, type_: str, fields: Sequence[str]) -> str:
    fields_list = "\n".join(f"    {field}" for field in fields)
    return f"fragment {name} on {type_} {{\n{fields_list}\n}}"
