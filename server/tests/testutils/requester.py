from typing import Any

from aiohttp import ClientResponse
from aiohttp.test_utils import TestClient
import pytest


class Requester:
    @pytest.fixture(scope="function", autouse=True)
    def setup(self, headers: dict, client: TestClient):
        self.headers = headers
        self.client = client

    @property
    def path(self) -> str:
        raise NotImplementedError()

    def build_path(self, **kwargs: Any) -> str:
        return self.path.format(**kwargs)

    async def get(self, assert_status: int = 200, **kwargs) -> ClientResponse:
        response = await self._request(method="GET", **kwargs)
        assert response.status == assert_status, response.status
        return response

    async def get_json(self, *args: Any, **kwargs: Any) -> Any:
        response = await self.get(*args, **kwargs)
        return await response.json()

    async def post(self, assert_status: int = 200, **kwargs) -> ClientResponse:
        response = await self._request(method="POST", **kwargs)
        assert response.status == assert_status, response.status
        return response

    async def post_json(self, *args, **kwargs) -> Any:
        response = await self.post(*args, **kwargs)
        return await response.json()

    async def delete(self, assert_status: int = 204, **kwargs) -> ClientResponse:
        response = await self._request(method="DELETE", **kwargs)
        assert response.status == assert_status, response.status
        return response

    async def put(self, assert_status: int = 200, **kwargs) -> ClientResponse:
        response = await self._request(method="PUT", **kwargs)
        assert response.status == assert_status, response.status
        return response

    async def put_json(self, *args, **kwargs) -> Any:
        response = await self.put(*args, **kwargs)
        return await response.json()

    async def _request(self, *, path_kwargs=None, **kwargs) -> ClientResponse:
        headers = kwargs.pop("headers", self.headers)
        path = self.build_path(**(path_kwargs or {}))
        return await self.client.request(path=path, headers=headers, **kwargs)
