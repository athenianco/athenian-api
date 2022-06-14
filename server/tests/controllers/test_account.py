import pytest
from sqlalchemy import delete, select

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.account import (
    copy_teams_as_needed,
    get_metadata_account_ids,
    get_user_account_status,
    match_metadata_installation,
)
from athenian.api.models.state.models import AccountGitHubAccount, Team
from athenian.api.models.web import User
from athenian.api.response import ResponseError
from tests.testutils.db import model_insert_stmt
from tests.testutils.factory.state import TeamFactory


@with_defer
async def test_get_metadata_account_id_cache(sdb, cache):
    assert await get_metadata_account_ids(1, sdb, cache) == (6366825,)
    await wait_deferred()
    assert len(cache.mem) == 1
    # use the cache
    assert await get_metadata_account_ids(1, sdb, cache) == (6366825,)
    await wait_deferred()
    assert len(cache.mem) == 1


async def test_get_metadata_account_id_conn(sdb):
    async with sdb.connection() as sdb_conn:
        assert await get_metadata_account_ids(1, sdb_conn, None) == (6366825,)


async def test_get_metadata_account_id_no_cache(sdb):
    assert await get_metadata_account_ids(1, sdb, None) == (6366825,)


async def test_get_metadata_account_id_error(sdb):
    with pytest.raises(ResponseError):
        await get_metadata_account_ids(2, sdb, None)


@with_defer
async def test_match_metadata_installation(sdb, mdb, slack):
    await sdb.execute(delete(AccountGitHubAccount))
    with pytest.raises(AssertionError):
        async with sdb.connection() as sdb_conn:
            async with sdb_conn.transaction():
                async with mdb.connection() as mdb_conn:
                    await match_metadata_installation(
                        1, "vmarkovtsev", sdb_conn, mdb_conn, mdb_conn, slack
                    )
    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            async with mdb.connection() as mdb_conn:
                meta_ids = await match_metadata_installation(
                    1, "vmarkovtsev", sdb_conn, mdb_conn, mdb, slack
                )
    assert meta_ids == {6366825}


async def test_copy_teams_as_needed(sdb: Database, mdb: Database):
    root_team_model = TeamFactory(name=Team.ROOT, parent_id=None)
    root_team_id = await sdb.execute(model_insert_stmt(root_team_model, with_primary_keys=False))

    created_teams, n = await copy_teams_as_needed(1, (6366825,), root_team_id, sdb, mdb, None)
    loaded_team_rows = await sdb.fetch_all(select(Team).where(Team.id != root_team_id))
    loaded_teams = {t[Team.name.name]: t for t in loaded_team_rows}

    assert len(created_teams) == len(loaded_teams) == n
    assert loaded_teams.keys() == {
        "team",
        "engineering",
        "business",
        "operations",
        "product",
        "admin",
        "automation",
    }
    assert loaded_teams["product"][Team.members.name] == [29, 39936]
    assert loaded_teams["product"][Team.parent_id.name] == loaded_teams["team"][Team.id.name]
    # team "team" hasn't a real parent team, so its parent team becomes root_team_id
    assert loaded_teams["team"][Team.parent_id.name] == root_team_id

    assert not any(team[Team.parent_id.name] is None for team in loaded_teams.values())

    created_teams, n = await copy_teams_as_needed(1, (6366825,), root_team_id, sdb, mdb, None)
    assert created_teams == []
    assert n == len(loaded_teams)


@with_defer
async def test_get_user_account_status_slack(sdb, mdb, slack):
    async def user():
        return User(login="gkwillie", email="bot@athenian.co")

    with pytest.raises(ResponseError):
        await get_user_account_status("github|60340680", 1, sdb, mdb, user, slack, None)
