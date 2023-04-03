import pytest

from athenian.api.db import Database
from athenian.api.internal.deployment_labels import DeploymentNotFoundError, get_deployment_labels
from tests.testutils.db import models_insert
from tests.testutils.factory.persistentdata import (
    DeployedLabelFactory,
    DeploymentNotificationFactory,
)


class TestGetDeploymentLabels:
    async def test_base(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d1"),
            DeploymentNotificationFactory(name="d2"),
            DeployedLabelFactory(deployment_name="d1", key="k1", value=123),
            DeployedLabelFactory(deployment_name="d1", key="k2", value={"foo": "bar"}),
            DeployedLabelFactory(deployment_name="d1", key="k3", value=None),
            DeployedLabelFactory(deployment_name="d1", key="k4", value="abc"),
            DeployedLabelFactory(deployment_name="d2", key="k0", value="def"),
        )
        res = await get_deployment_labels("d1", 1, rdb)
        assert res == {"k1": 123, "k2": {"foo": "bar"}, "k3": None, "k4": "abc"}

    async def test_no_labels(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d"))
        assert await get_deployment_labels("d", 1, rdb) == {}

    async def test_not_found(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d"))
        with pytest.raises(DeploymentNotFoundError):
            await get_deployment_labels("d2", 1, rdb)

        with pytest.raises(DeploymentNotFoundError):
            await get_deployment_labels("d", 2, rdb)
