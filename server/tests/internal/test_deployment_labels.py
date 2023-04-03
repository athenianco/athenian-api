from operator import itemgetter

import pytest

from athenian.api.db import Database, integrity_errors
from athenian.api.internal.deployment_labels import (
    DeploymentNotFoundError,
    delete_deployment_labels,
    get_deployment_labels,
    lock_deployment,
    upsert_deployment_labels,
)
from athenian.api.models.persistentdata.models import DeployedLabel, DeploymentNotification
from tests.testutils.db import (
    assert_existing_row,
    assert_existing_rows,
    assert_missing_row,
    models_insert,
    transaction_conn,
)
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


class TestLockDeployment:
    async def test_not_found(self, rdb: Database) -> None:
        async with transaction_conn(rdb) as rdb_conn:
            with pytest.raises(DeploymentNotFoundError):
                await lock_deployment("d", 1, rdb_conn)

    async def test_found(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d"))
        async with transaction_conn(rdb) as rdb_conn:
            await lock_deployment("d", 1, rdb_conn)


class TestUpsertDeploymentLabels:
    async def test_smoke(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d1"),
            DeployedLabelFactory(deployment_name="d1", key="k1", value=123),
            DeployedLabelFactory(deployment_name="d1", key="k2", value="a"),
        )
        await upsert_deployment_labels("d1", 1, {"k2": "b", "k3": ["c"]}, rdb)
        await assert_existing_row(rdb, DeploymentNotification, name="d1")
        labels = await assert_existing_rows(rdb, DeployedLabel, deployment_name="d1")
        labels = sorted(labels, key=itemgetter(DeployedLabel.key.name))
        assert len(labels) == 3

        assert [label[DeployedLabel.key.name] for label in labels] == ["k1", "k2", "k3"]
        assert [label[DeployedLabel.value.name] for label in labels] == [123, "b", ["c"]]

    @pytest.mark.xfail(reason="missing foreign key constraint in DeployedLabel models")
    async def test_deployment_not_existing(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
        )
        with pytest.raises(integrity_errors):
            await upsert_deployment_labels("d", 2, {"k": "v"}, rdb)


class TestDeleteDeploymentLabels:
    async def test_smoke(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d1"),
            DeployedLabelFactory(deployment_name="d1", key="k1", value=123),
            DeployedLabelFactory(deployment_name="d1", key="k2"),
            DeploymentNotificationFactory(name="d2"),
            DeployedLabelFactory(deployment_name="d2", key="k3", value=3),
        )
        await delete_deployment_labels("d1", 1, ["k3", "k2"], rdb)

        await assert_existing_row(rdb, DeploymentNotification, name="d1")
        await assert_existing_rows(rdb, DeployedLabel, deployment_name="d1", key="k1", value=123)
        await assert_missing_row(rdb, DeployedLabel, deployment_name="d1", key="k2")
        await assert_missing_row(rdb, DeployedLabel, deployment_name="d1", key="k3")

        await assert_existing_row(rdb, DeploymentNotification, name="d2")
        await assert_existing_rows(rdb, DeployedLabel, deployment_name="d2", key="k3", value=3)

    async def test_different_account(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
            DeployedLabelFactory(deployment_name="d", key="k1", value=123),
        )
        await delete_deployment_labels("d", 2, ["k1"], rdb)
        await assert_existing_row(rdb, DeploymentNotification, name="d")
        await assert_existing_rows(rdb, DeployedLabel, deployment_name="d", key="k1", value=123)
