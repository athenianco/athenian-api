from typing import Any

from aiohttp import ClientResponse
import pytest

from athenian.api.db import Database
from athenian.api.defer import wait_all_deferred
from athenian.api.models.persistentdata.models import DeployedLabel
from athenian.api.models.precomputed.models import GitHubDeploymentFacts
from tests.testutils.db import DBCleaner, assert_existing_rows, assert_missing_row, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.persistentdata import (
    DeployedComponentFactory,
    DeployedLabelFactory,
    DeploymentNotificationFactory,
)
from tests.testutils.factory.precomputed import GitHubDeploymentFactsFactory
from tests.testutils.factory.state import UserAccountFactory, UserTokenFactory
from tests.testutils.factory.wizards import insert_logical_repo, insert_repo
from tests.testutils.requester import Requester
from tests.testutils.time import dt


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


class BaseModifyDeploymentLabelsTest(Requester):
    path = "/v1/events/deployment/{name}/labels"
    _TOKEN_ID = 1

    @pytest.fixture(scope="function", autouse=True)
    async def setup_users(self, sdb):
        await models_insert(
            sdb,
            UserAccountFactory(user_id="auth0|XXX"),
            UserTokenFactory(user_id="auth0|XXX", id=self._TOKEN_ID),
        )

    async def patch(self, name: str, *args: Any, **kwargs: Any) -> ClientResponse:
        path_kwargs = {"name": name}
        if "token" not in kwargs:
            kwargs["token"] = self.encode_token(self._TOKEN_ID)
        return await super().patch(*args, path_kwargs=path_kwargs, **kwargs)

    async def _assert_db_labels(self, name: str, rdb: Database, **labels) -> None:
        if not labels:
            await assert_missing_row(rdb, DeployedLabel, deployment_name=name)
            return

        rows = await assert_existing_rows(rdb, DeployedLabel, deployment_name=name)
        assert len(rows) == len(labels)
        for r in rows:
            value = labels.pop(r[DeployedLabel.key.name])
            assert value == r[DeployedLabel.value.name]


class TestModifyDeploymentLabelsErrors(BaseModifyDeploymentLabelsTest):
    async def test_unauthorized(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d0"))
        await self.patch_json("d0", token=None, json={"upsert": {"a": "b"}}, assert_status=401)
        await assert_missing_row(rdb, DeployedLabel, deployment_name="d0")

    async def test_default_user(self, rdb: Database, sdb: Database) -> None:
        await models_insert(sdb, UserTokenFactory(user_id="auth0|62a1ae88b6bba16c6dbc6870", id=2))
        await models_insert(rdb, DeploymentNotificationFactory(name="d"))
        body = {"upsert": {"a": "b"}}
        await self.patch_json("d", token=self.encode_token(2), json=body, assert_status=403)
        await assert_missing_row(rdb, DeployedLabel, deployment_name="d")

    async def test_not_found(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d0"))
        await self.patch_json("d1", json={"upsert": {"a": "b"}}, assert_status=404)
        await assert_missing_row(rdb, DeployedLabel, deployment_name="d0")

    async def test_account_mismatch(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(account_id=4, name="d0"))
        await self.patch_json("d0", json={"upsert": {"a": "b"}}, assert_status=404)
        await assert_missing_row(rdb, DeployedLabel, deployment_name="d0")

    async def test_label_both_in_upsert_and_delete(self, rdb: Database) -> None:
        await models_insert(rdb, DeploymentNotificationFactory(name="d"))
        body = {"upsert": {"a": "b"}, "delete": ["a"]}
        res = await self.patch_json("d", json=body, assert_status=400)
        assert res["detail"] == 'Keys cannot appear both in "delete" and "upsert": a'
        await assert_missing_row(rdb, DeployedLabel, deployment_name="d")


class TestModifyDeploymentLabels(BaseModifyDeploymentLabelsTest):
    async def test_upsert(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
            DeployedLabelFactory(deployment_name="d", key="k0", value="v0"),
        )
        res = await self.patch_json("d", json={"upsert": {"k0": True, "k1": {"foo": {}}}})
        assert res == {"labels": {"k0": True, "k1": {"foo": {}}}}
        await self._assert_db_labels("d", rdb, k0=True, k1={"foo": {}})

    async def test_delete(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
            DeployedLabelFactory(deployment_name="d", key="k0", value="v0"),
            DeployedLabelFactory(deployment_name="d", key="k1", value="v1"),
            DeployedLabelFactory(deployment_name="d", key="k2", value="v2"),
        )
        res = await self.patch_json("d", json={"delete": ["k0", "k2"]})
        assert res == {"labels": {"k1": "v1"}}
        await self._assert_db_labels("d", rdb, k1="v1")

        res = await self.patch_json("d", json={"delete": ["k1"]})
        assert res == {"labels": {}}
        await self._assert_db_labels("d", rdb)

    async def test_delete_unknown_labels(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
            DeployedLabelFactory(deployment_name="d", key="k0", value="v0"),
        )
        res = await self.patch_json("d", json={"delete": ["k1"]})
        assert res == {"labels": {"k0": "v0"}}
        await self._assert_db_labels("d", rdb, k0="v0")

    async def test_upsert_and_delete(self, rdb: Database) -> None:
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
            DeployedLabelFactory(deployment_name="d", key="k0", value="v0"),
            DeployedLabelFactory(deployment_name="d", key="k1", value="v1"),
        )
        body = {"upsert": {"k1": ["l0"], "k2": "bar"}, "delete": ["k0"]}
        res = await self.patch_json("d", json=body)
        assert res == {"labels": {"k1": ["l0"], "k2": "bar"}}
        await self._assert_db_labels("d", rdb, k1=["l0"], k2="bar")


class TestModifyDeploymentLabelsPDBInvalidation(BaseModifyDeploymentLabelsTest):
    async def test_base(
        self,
        rdb: Database,
        sdb: Database,
        pdb: Database,
        mdb_rw: Database,
    ) -> None:
        await models_insert(
            pdb,
            GitHubDeploymentFactsFactory(deployment_name="d"),
            GitHubDeploymentFactsFactory(deployment_name="d2"),
        )
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d", finished_at=dt(2021, 1, 2)),
            DeployedComponentFactory(deployment_name="d", repository_node_id=99),
            DeployedLabelFactory(deployment_name="d", key="k", value="v"),
            DeploymentNotificationFactory(name="d2", finished_at=dt(2021, 1, 3)),
            DeployedComponentFactory(deployment_name="d2", repository_node_id=99),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            await insert_logical_repo(99, "l", sdb, deployments={"labels": {"k": "v"}})

            body = {"delete": ["k"]}
            res = await self.patch_json("d", json=body)
            assert res == {"labels": {}}

            await self._assert_db_labels("d", rdb)

            await wait_all_deferred()

            await assert_missing_row(pdb, GitHubDeploymentFacts, deployment_name="d")
            await assert_missing_row(pdb, GitHubDeploymentFacts, deployment_name="d1")
