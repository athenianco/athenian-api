from typing import Any

from aiohttp.test_utils import TestClient
import pytest


class Requester:
    @pytest.fixture(scope="function", autouse=True)
    def setup(self, headers, client):
        self.headers: dict = headers
        self.client: TestClient = client

    async def _request(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
