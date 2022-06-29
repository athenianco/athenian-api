from datetime import datetime, timezone

from freezegun import freeze_time
import pytest
from sqlalchemy import delete, distinct, insert, select, update
from sqlalchemy.sql.functions import count

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.jira import (
    disable_empty_projects,
    load_jira_identity_mapping_sentinel,
    load_mapped_jira_users,
    match_jira_identities,
    normalize_issue_type,
)
from athenian.api.models.metadata.jira import Progress
from athenian.api.models.state.models import (
    AccountJiraInstallation,
    JIRAProjectSetting,
    MappedJIRAIdentity,
)
from tests.testutils.db import DBCleaner, assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import (
    AccountFactory,
    AccountGitHubAccountFactory,
    AccountJiraInstallationFactory,
)
from tests.testutils.time import dt


@with_defer
async def test_load_mapped_jira_users_cache(sdb, mdb, cache):
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {}
    await wait_deferred()
    await sdb.execute(
        insert(MappedJIRAIdentity).values(
            MappedJIRAIdentity(
                account_id=1,
                github_user_id=40020,
                jira_user_id="5de5049e2c5dd20d0f9040c1",
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {}
    await wait_deferred()
    await load_jira_identity_mapping_sentinel.reset_cache(1, cache)
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {40020: "Vadim Markovtsev"}


@with_defer
async def test_load_mapped_jira_users_installation(sdb, mdb, cache):
    await sdb.execute(delete(AccountJiraInstallation))
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {}
    await wait_deferred()
    await sdb.execute(
        insert(AccountJiraInstallation).values(
            {
                AccountJiraInstallation.account_id: 1,
                AccountJiraInstallation.id: 1,
            },
        ),
    )
    await sdb.execute(
        insert(MappedJIRAIdentity).values(
            MappedJIRAIdentity(
                account_id=1,
                github_user_id=40020,
                jira_user_id="5de5049e2c5dd20d0f9040c1",
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {40020: "Vadim Markovtsev"}


async def test_load_mapped_jira_users_empty(sdb, mdb):
    mapping = await load_mapped_jira_users(1, [], sdb, mdb, None)
    assert mapping == {}


async def test_load_mapped_jira_users_no_jira(sdb, mdb):
    await sdb.execute(
        insert(MappedJIRAIdentity).values(
            MappedJIRAIdentity(
                account_id=1,
                github_user_id=40020,
                jira_user_id="5de5049e2c5dd20d0f9040c1",
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    await sdb.execute(delete(AccountJiraInstallation))
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, None)
    assert mapping == {}


async def test_match_jira_identities_from_scratch(sdb, mdb, slack):
    matched = await match_jira_identities(1, (6366825,), sdb, mdb, slack, None)
    assert matched == 9
    stored = await sdb.fetch_all(
        select(
            [
                MappedJIRAIdentity.github_user_id,
                MappedJIRAIdentity.jira_user_id,
                MappedJIRAIdentity.confidence,
            ],
        ),
    )
    assert matched == len(stored)
    github_users = set()
    jira_users = set()
    for row in stored:
        if row[MappedJIRAIdentity.github_user_id.name] != 58:
            assert row[MappedJIRAIdentity.confidence.name] == 1
        else:
            assert row[MappedJIRAIdentity.confidence.name] == 0.75
        github_users.add(row[MappedJIRAIdentity.github_user_id.name])
        jira_users.add(row[MappedJIRAIdentity.jira_user_id.name])
    assert len(github_users) == len(jira_users) == matched


async def test_match_jira_identities_incremental(sdb, mdb, slack):
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
    matched = await match_jira_identities(1, (6366825,), sdb, mdb, slack, None)
    assert matched == 8
    stored = await sdb.fetch_val(select([count(distinct(MappedJIRAIdentity.github_user_id))]))
    assert matched + 1 == stored


@pytest.mark.flaky(reruns=3)
async def test_match_jira_identities_incomplete_progress(sdb, mdb_rw, slack):
    await mdb_rw.execute(
        insert(Progress).values(
            {
                Progress.current.name: 1,
                Progress.total.name: 2,
                Progress.acc_id.name: 1,
                Progress.event_id.name: "guid",
                Progress.event_type.name: "user",
                Progress.started_at.name: datetime.now(timezone.utc),
                Progress.end_at.name: datetime.now(timezone.utc),
                Progress.is_initial.name: True,
            },
        ),
    )
    try:
        assert (await match_jira_identities(1, (6366825,), sdb, mdb_rw, slack, None)) is None
    finally:
        await mdb_rw.execute(update(Progress).values({Progress.current.name: 10}))


@pytest.mark.parametrize(
    "orig, norm",
    [
        ("Súb-tasks", "subtask"),
        ("Stories", "story"),
        ("", ""),
        ("BUG", "bug"),
    ],
)
def test_normalize_issue_type(orig, norm):
    assert normalize_issue_type(orig) == norm


class TestDisableEmptyProjects:
    @with_defer
    async def test_base(self, sdb, mdb, slack, cache):
        disabled = await disable_empty_projects(1, (6366825,), sdb, mdb, slack, cache)
        assert disabled == 1
        settings = await sdb.fetch_all(
            select([JIRAProjectSetting.key, JIRAProjectSetting.enabled]).where(
                JIRAProjectSetting.account_id == 1,
            ),
        )
        assert len(settings) == 1
        keys = set()
        for row in settings:
            keys.add(row[JIRAProjectSetting.key.name])
            assert not row[JIRAProjectSetting.enabled.name]
        assert keys == {"ENG"}

    @freeze_time("2020-01-01")
    async def test_duplicate_proj_keys(self, sdb: Database, mdb: Database) -> None:
        ACC_ID = 4
        MD_ACC_ID = 104
        JIRA_INST_ID = 204
        async with DBCleaner(mdb) as mdb_cleaner:
            await models_insert(
                sdb,
                AccountFactory(id=ACC_ID),
                AccountGitHubAccountFactory(account_id=ACC_ID, id=MD_ACC_ID),
                AccountJiraInstallationFactory(account_id=ACC_ID, id=JIRA_INST_ID),
            )
            old_dt = dt(2019, 1, 1)
            md_models = [
                md_factory.AccountFactory(id=MD_ACC_ID),
                md_factory.JIRAProjectFactory(acc_id=JIRA_INST_ID, id="101", key="PRJ1"),
                md_factory.JIRAProjectFactory(acc_id=JIRA_INST_ID, id="111", key="PRJ1"),
                md_factory.JIRAProjectFactory(acc_id=JIRA_INST_ID, id="102", key="PRJ2"),
                md_factory.JIRAProjectFactory(acc_id=JIRA_INST_ID, id="122", key="PRJ2"),
                md_factory.JIRAIssueFactory(acc_id=JIRA_INST_ID, project_id="101", created=old_dt),
                md_factory.JIRAIssueFactory(acc_id=JIRA_INST_ID, project_id="111", created=old_dt),
                md_factory.JIRAIssueFactory(acc_id=JIRA_INST_ID, project_id="102", created=old_dt),
            ]
            mdb_cleaner.add_models(*md_models)
            await models_insert(mdb, *md_models)

            n_disabled = await disable_empty_projects(ACC_ID, (MD_ACC_ID,), sdb, mdb, None, None)

        # PRJ1 key must be disabled since both 101 and 111 are to be disabled (old first issue)
        # PRJ2 key must not be disabled since 122 is not to be disabled (no issues)
        assert n_disabled == 1
        await assert_existing_row(sdb, JIRAProjectSetting, key="PRJ1", enabled=False)
        await assert_missing_row(sdb, JIRAProjectSetting, key="PRJ2", enabled=False)
