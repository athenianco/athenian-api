from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.internal.miners.types import PullRequestFacts
from athenian.api.models.metadata.github import Branch
from athenian.api.models.persistentdata.models import DeployedLabel
from athenian.api.models.state.models import (
    AccountJiraInstallation,
    JIRAProjectSetting,
    MappedJIRAIdentity,
    RepositorySet,
    Team,
)
from athenian.api.typing_utils import wraps


@pytest.fixture(scope="function")
async def disabled_dev(sdb):
    await sdb.execute(
        insert(JIRAProjectSetting).values(
            JIRAProjectSetting(account_id=1, key="DEV", enabled=False)
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def dummy_deployment_label(rdb):
    await rdb.execute(
        insert(DeployedLabel).values(
            DeployedLabel(
                account_id=1,
                deployment_name="Dummy deployment",
                key="xxx",
                value=["yyy"],
            ).explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def logical_reposet(sdb):
    await sdb.execute(
        update(RepositorySet)
        .where(RepositorySet.id == 1)
        .values(
            {
                RepositorySet.items: [
                    ["github.com", 40550, ""],
                    ["github.com", 40550, "alpha"],
                    ["github.com", 40550, "beta"],
                    ["github.com", 39652769, ""],
                ],
                RepositorySet.updated_at: datetime.now(timezone.utc),
                RepositorySet.updates_count: 2,
            },
        ),
    )


@pytest.fixture(scope="function")
async def vadim_id_mapping(sdb):
    await sdb.execute(
        insert(MappedJIRAIdentity).values(
            MappedJIRAIdentity(
                account_id=1,
                github_user_id=40020,
                jira_user_id="5de5049e2c5dd20d0f9040c1",
                confidence=1.0,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def denys_id_mapping(sdb):
    await sdb.execute(
        insert(MappedJIRAIdentity).values(
            MappedJIRAIdentity(
                account_id=1,
                github_user_id=40294,
                jira_user_id="5de4cff936b8050e29258600",
                confidence=1.0,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def no_jira(sdb):
    await sdb.execute(delete(AccountJiraInstallation))


class FakeFacts(PullRequestFacts):
    def __init__(self):
        super().__init__(b"\0" * PullRequestFacts.dtype.itemsize)


def with_only_master_branch(func):
    async def wrapped_with_only_master_branch(**kwargs):
        mdb = kwargs["mdb_rw"]
        branches = await mdb.fetch_all(select(Branch).where(Branch.branch_name != "master"))
        await mdb.execute(delete(Branch).where(Branch.branch_name != "master"))
        try:
            await func(**kwargs)
        finally:
            for branch in branches:
                await mdb.execute(insert(Branch).values(branch))

    return wraps(wrapped_with_only_master_branch, func)


@pytest.fixture(scope="function")
async def sample_team(sdb):
    return await sdb.execute(
        insert(Team).values(
            Team(
                owner_id=1,
                name="Sample",
                members=[51, 40020, 39789, 40070],
                parent_id=None,
            )
            .create_defaults()
            .explode(),
        ),
    )
