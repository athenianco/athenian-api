import pytest
from sqlalchemy import delete, distinct, insert, select, update
from sqlalchemy.sql.functions import count

from athenian.api.controllers.jira import load_jira_identity_mapping_sentinel, \
    load_mapped_jira_users, match_jira_identities
from athenian.api.defer import with_defer
from athenian.api.models.metadata.jira import Progress
from athenian.api.models.state.models import AccountJiraInstallation, MappedJIRAIdentity


@with_defer
async def test_load_mapped_jira_users_cache(sdb, mdb, cache):
    mapping = await load_mapped_jira_users(1, ["MDQ6VXNlcjI3OTM1NTE="], sdb, mdb, cache)
    assert mapping == {}
    await sdb.execute(insert(MappedJIRAIdentity).values(MappedJIRAIdentity(
        account_id=1,
        github_user_id="MDQ6VXNlcjI3OTM1NTE=",
        jira_user_id="5de5049e2c5dd20d0f9040c1",
    ).create_defaults().explode(with_primary_keys=True)))
    mapping = await load_mapped_jira_users(1, ["MDQ6VXNlcjI3OTM1NTE="], sdb, mdb, cache)
    assert mapping == {}
    await load_jira_identity_mapping_sentinel.reset_cache(1, cache)
    mapping = await load_mapped_jira_users(1, ["MDQ6VXNlcjI3OTM1NTE="], sdb, mdb, cache)
    assert mapping == {"MDQ6VXNlcjI3OTM1NTE=": "Vadim Markovtsev"}


async def test_load_mapped_jira_users_empty(sdb, mdb):
    mapping = await load_mapped_jira_users(1, [], sdb, mdb, None)
    assert mapping == {}


async def test_load_mapped_jira_users_no_jira(sdb, mdb):
    await sdb.execute(insert(MappedJIRAIdentity).values(MappedJIRAIdentity(
        account_id=1,
        github_user_id="MDQ6VXNlcjI3OTM1NTE=",
        jira_user_id="5de5049e2c5dd20d0f9040c1",
    ).create_defaults().explode(with_primary_keys=True)))
    await sdb.execute(delete(AccountJiraInstallation))
    mapping = await load_mapped_jira_users(1, ["MDQ6VXNlcjI3OTM1NTE="], sdb, mdb, None)
    assert mapping == {}


async def test_match_jira_identities_from_scratch(sdb, mdb, slack):
    matched = await match_jira_identities(1, (6366825,), sdb, mdb, slack, None)
    assert matched == 5
    stored = await sdb.fetch_all(select([MappedJIRAIdentity.github_user_id,
                                         MappedJIRAIdentity.jira_user_id,
                                         MappedJIRAIdentity.confidence]))
    assert matched == len(stored)
    github_users = set()
    jira_users = set()
    for row in stored:
        assert row[MappedJIRAIdentity.confidence.key] == 1
        github_users.add(row[MappedJIRAIdentity.github_user_id.key])
        jira_users.add(row[MappedJIRAIdentity.jira_user_id.key])
    assert len(github_users) == len(jira_users) == matched


async def test_match_jira_identities_incremental(sdb, mdb, slack):
    await sdb.execute(insert(MappedJIRAIdentity).values(
        MappedJIRAIdentity(
            account_id=1,
            github_user_id="MDQ6VXNlcjY3NjcyNA==",
            jira_user_id="5de4cff936b8050e29258600",
            confidence=1.0,
        ).create_defaults().explode(with_primary_keys=True),
    ))
    matched = await match_jira_identities(1, (6366825,), sdb, mdb, slack, None)
    assert matched == 4
    stored = await sdb.fetch_val(select([count(distinct(MappedJIRAIdentity.github_user_id))]))
    assert matched + 1 == stored


@pytest.mark.flaky(reruns=3)
async def test_match_jira_identities_incomplete_progress(sdb, mdb, slack):
    await mdb.execute(update(Progress).values({Progress.current.key: 1}))
    try:
        assert (await match_jira_identities(1, (6366825,), sdb, mdb, slack, None)) is None
    finally:
        await mdb.execute(update(Progress).values({Progress.current.key: 10}))
