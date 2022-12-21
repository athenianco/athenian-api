from typing import Any

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

    async def get(self, assert_status: int = 200, **kwargs) -> Any:
        response = await self._request(method="GET", **kwargs)
        assert response.status == assert_status, response.status
        return response

    async def get_json(self, *args: Any, **kwargs: Any) -> Any:
        response = await self.get(*args, **kwargs)
        return await response.json()

    async def post(self, assert_status: int = 200, **kwargs) -> Any:
        response = await self._request(method="POST", **kwargs)
        assert response.status == assert_status, response.status
        return response

    async def post_json(self, *args, **kwargs) -> Any:
        response = await self.post(*args, **kwargs)
        return await response.json()

    async def _request(self, *, path_kwargs=None, **kwargs) -> Any:
        path = self.build_path(**(path_kwargs or {}))
        return await self.client.request(path=path, headers=self.headers, **kwargs)
