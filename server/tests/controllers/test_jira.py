from sqlalchemy import delete, insert

from athenian.api.controllers.jira import load_jira_identity_mapping_sentinel, \
    load_mapped_jira_users
from athenian.api.defer import with_defer
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
