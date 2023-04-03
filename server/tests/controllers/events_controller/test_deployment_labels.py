from typing import Any

from aiohttp import ClientResponse

from athenian.api.db import Database
from tests.testutils.db import models_insert
from tests.testutils.factory.persistentdata import (
    DeployedLabelFactory,
    DeploymentNotificationFactory,
)
from tests.testutils.requester import Requester


class BaseGetDeploymentLabelsTest(Requester):
    path = "/v1/events/deployment/{name}/labels"

    async def get(self, name: str, *args: Any, **kwargs: Any) -> ClientResponse:
        path_kwargs = {"name": name}
        return await super().get(*args, path_kwargs=path_kwargs, **kwargs)


class TestGetDeploymentLabelsErrors(BaseGetDeploymentLabelsTest):
    async def test_unauthorized(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d0"))
        await self.get_json("d0", assert_status=401)

    async def test_not_found(self, rdb: Database, token: str) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d0"))
        await self.get_json("d1", token=token, assert_status=404)

    async def test_account_mismatch(self, rdb: Database, token: str) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(account_id=4, name="d0"))
        await self.get_json("d0", token=token, assert_status=404)


class TestGetDeploymentLabels(BaseGetDeploymentLabelsTest):
    async def test_no_labels(self, rdb: Database, token: str) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d"))
        res = await self.get_json("d", token=token)
        assert res == {"labels": {}}

    async def test_base(self, rdb: Database, token: str) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
            DeployedLabelFactory(deployment_name="d", key="k0", value=["DEV-123"]),
            DeployedLabelFactory(deployment_name="d", key="k1", value=42),
        )
        res = await self.get_json("d", token=token)
        assert res == {"labels": {"k0": ["DEV-123"], "k1": 42}}
