import pytest
from sqlalchemy import delete, distinct, insert, select, update
from sqlalchemy.sql.functions import count

from athenian.api.controllers.jira import load_jira_identity_mapping_sentinel, \
    load_mapped_jira_users, match_jira_identities, normalize_issue_type
from athenian.api.defer import with_defer
from athenian.api.models.metadata.jira import Progress
from athenian.api.models.state.models import AccountJiraInstallation, MappedJIRAIdentity


@with_defer
async def test_load_mapped_jira_users_cache(sdb, mdb, cache):
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {}
    await sdb.execute(insert(MappedJIRAIdentity).values(MappedJIRAIdentity(
        account_id=1,
        github_user_id=40020,
        jira_user_id="5de5049e2c5dd20d0f9040c1",
    ).create_defaults().explode(with_primary_keys=True)))
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {}
    await load_jira_identity_mapping_sentinel.reset_cache(1, cache)
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, cache)
    assert mapping == {40020: "Vadim Markovtsev"}


async def test_load_mapped_jira_users_empty(sdb, mdb):
    mapping = await load_mapped_jira_users(1, [], sdb, mdb, None)
    assert mapping == {}


async def test_load_mapped_jira_users_no_jira(sdb, mdb):
    await sdb.execute(insert(MappedJIRAIdentity).values(MappedJIRAIdentity(
        account_id=1,
        github_user_id=40020,
        jira_user_id="5de5049e2c5dd20d0f9040c1",
    ).create_defaults().explode(with_primary_keys=True)))
    await sdb.execute(delete(AccountJiraInstallation))
    mapping = await load_mapped_jira_users(1, [40020], sdb, mdb, None)
    assert mapping == {}


async def test_match_jira_identities_from_scratch(sdb, mdb, slack):
    matched = await match_jira_identities(1, (6366825,), sdb, mdb, slack, None)
    assert matched == 9
    stored = await sdb.fetch_all(select([MappedJIRAIdentity.github_user_id,
                                         MappedJIRAIdentity.jira_user_id,
                                         MappedJIRAIdentity.confidence]))
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
    await sdb.execute(insert(MappedJIRAIdentity).values(
        MappedJIRAIdentity(
            account_id=1,
            github_user_id=40294,
            jira_user_id="5de4cff936b8050e29258600",
            confidence=1.0,
        ).create_defaults().explode(with_primary_keys=True),
    ))
    matched = await match_jira_identities(1, (6366825,), sdb, mdb, slack, None)
    assert matched == 8
    stored = await sdb.fetch_val(select([count(distinct(MappedJIRAIdentity.github_user_id))]))
    assert matched + 1 == stored


@pytest.mark.flaky(reruns=3)
async def test_match_jira_identities_incomplete_progress(sdb, mdb_rw, slack):
    await mdb_rw.execute(update(Progress).values({Progress.current.name: 1}))
    try:
        assert (await match_jira_identities(1, (6366825,), sdb, mdb_rw, slack, None)) is None
    finally:
        await mdb_rw.execute(update(Progress).values({Progress.current.name: 10}))


@pytest.mark.parametrize("orig, norm", [
    ("Súb-tasks", "subtask"),
    ("Stories", "story"),
    ("", ""),
    ("BUG", "bug"),
])
def test_normalize_issue_type(orig, norm):
    assert normalize_issue_type(orig) == norm
