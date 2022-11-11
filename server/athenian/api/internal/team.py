from datetime import datetime, timezone
from http import HTTPStatus
from itertools import chain
import logging
from typing import Collection, Iterable, Optional, Sequence

import aiomcache
import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.align.goals.dbaccess import delete_empty_goals
from athenian.api.async_utils import gather
from athenian.api.db import Connection, Database, DatabaseLike, Row, conn_in_transaction
from athenian.api.internal.jira import load_mapped_jira_users
from athenian.api.models.metadata.github import User
from athenian.api.models.state.models import Team, UserAccount
from athenian.api.models.web import BadRequestError, Contributor, GenericError
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span


class TeamNotFoundError(ResponseError):
    """A team was not found."""

    def __init__(self, *team_id: int):
        """Init the TeamNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/teams/TeamNotFound",
            status=HTTPStatus.NOT_FOUND,
            detail=(
                f"Team{'s' if len(team_id) > 1 else ''} {', '.join(map(str, team_id))} "
                "not found or access denied"
            ),
            title="Team not found",
        )
        super().__init__(wrapped_error)


class RootTeamNotFoundError(ResponseError):
    """A team was not found."""

    def __init__(self):
        """Init the RootTeamNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/teams/TeamNotFound",
            status=HTTPStatus.NOT_FOUND,
            detail="Root team not found or access denied",
            title="Root team not found",
        )
        super().__init__(wrapped_error)


class MultipleRootTeamsError(ResponseError):
    """An account has multiple root teams."""

    def __init__(self, account_id: int):
        """Init the MultipleRootTeamsError."""
        wrapped_error = GenericError(
            type="/errors/teams/MultipleRootTeamsError",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Account {account_id} has multiple root teams",
            title="Multiple root teams",
        )
        super().__init__(wrapped_error)


@sentry_span
async def get_root_team(account_id: int, sdb_conn: DatabaseLike) -> Row:
    """Return the root team for the account."""
    stmt = sa.select(Team).where(sa.and_(Team.owner_id == account_id, Team.parent_id.is_(None)))
    root_teams = await sdb_conn.fetch_all(stmt)
    if not root_teams:
        raise RootTeamNotFoundError()
    if len(root_teams) > 1:
        raise MultipleRootTeamsError(account_id)
    return root_teams[0]


@sentry_span
async def get_team_from_db(
    team_id: int,
    account_id: Optional[int],
    user_id: Optional[str],
    sdb_conn: DatabaseLike,
) -> Row:
    """Return a team owned by an account.

    Team can be filtered by owner account and/or user id with access to team.
    At least one of the two filtering conditions must be specified.

    """
    if account_id is None and user_id is None:
        raise ValueError("At least one between account_id and user_id must be specified")

    where = Team.id == team_id
    if account_id is not None:
        where = sa.and_(where, Team.owner_id == account_id)
        from_ = Team
    if user_id is not None:
        from_ = sa.join(Team, UserAccount, Team.owner_id == UserAccount.account_id)
        where = sa.and_(where, UserAccount.user_id == user_id)

    stmt = sa.select(Team).select_from(from_).where(where)
    team = await sdb_conn.fetch_one(stmt)
    if team is None:
        raise TeamNotFoundError(team_id)
    return team


@sentry_span
async def get_all_team_members(
    gh_user_ids: Iterable[int],
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    sdb: Database,
    cache: Optional[aiomcache.Client],
) -> dict[int, Contributor]:
    """Return contributor objects for given github user identifiers."""
    user_rows, mapped_jira = await gather(
        mdb.fetch_all(
            sa.select(User).where(
                User.acc_id.in_(meta_ids),
                User.node_id.in_(gh_user_ids),
                User.login.isnot(None),
            ),
        ),
        load_mapped_jira_users(account, gh_user_ids, sdb, mdb, cache),
    )
    user_by_node = {u[User.node_id.name]: u for u in user_rows}
    all_contributors = {}
    missing = []
    for m in gh_user_ids:
        try:
            ud = user_by_node[m]
        except KeyError:
            missing.append(m)
            continue
        else:
            login = ud[User.html_url.name].split("://", 1)[1]
            c = Contributor(
                login=login,
                name=ud[User.name.name],
                email=ud[User.email.name],
                picture=ud[User.avatar_url.name],
                jira_user=mapped_jira.get(m),
            )
        all_contributors[m] = c

    if missing:
        logging.getLogger("team.get_all_team_members").error(
            "Some users are missing in %s: %s", meta_ids, missing,
        )
    return all_contributors


@sentry_span
async def fetch_teams_recursively(
    account: int,
    sdb: DatabaseLike,
    select_entities: Sequence[InstrumentedAttribute] = (Team.id,),
    root_team_ids: Optional[Collection] = None,
    max_depth: int = None,
) -> Sequence[Row]:
    """Return the recursively collected list of teams for the account.

    If `root_team_ids` is passed those will be taken as base for the recursion.
    If `root_team_ids` is None all root teams will be taken as base for the recursion.

    The returned list of teams will include duplicates when
    one of the `root_team_ids` is ancestor of another.

    Returned columns can be selected with `select_entities`. The ID of the root team
    used to fetch the row is always included as the last column.
    """
    # a recursive CTE is used to link children teams and track depth

    # Team.id is required inside cte for join even if not requested by caller
    if Team.id not in select_entities:
        cte_select_entities: Sequence[InstrumentedAttribute] = (*select_entities, Team.id)
    else:
        cte_select_entities = select_entities

    # base team is the specified root_team_id if present, else all account's root teams
    recursive_base_where = Team.owner_id == account
    if root_team_ids is None:
        recursive_base_where = sa.and_(recursive_base_where, Team.parent_id.is_(None))
    else:
        recursive_base_where = sa.and_(recursive_base_where, Team.id.in_(root_team_ids))

    recursive_base = sa.select(
        *cte_select_entities,
        sa.cast(1, sa.Integer).label("depth"),
        Team.id.label(Team.root_id),
    ).where(recursive_base_where)
    cte = recursive_base.cte("teams_cte", recursive=True)
    recursive_step_where = sa.and_(Team.parent_id == cte.c.id, Team.owner_id == account)
    # stop recursion on depth if requested by caller
    if max_depth is not None:
        recursive_step_where = sa.and_(recursive_step_where, cte.c.depth < max_depth)
    recursive_step = sa.select(*cte_select_entities, cte.c.depth + 1, cte.c.root_id).where(
        recursive_step_where,
    )
    cte = cte.union_all(recursive_step)

    # selected entities have the same names but must be taken from cte selectable
    result_select_entities = (getattr(cte.c, entity.name) for entity in select_entities)
    stmt = sa.select(*result_select_entities, cte.c.root_id)

    rows = await sdb.fetch_all(stmt)
    if missing := set(root_team_ids or set()) - {r[Team.root_id] for r in rows}:
        raise TeamNotFoundError(*missing)
    return rows


async def fetch_team_members_recursively(
    account: int,
    sdb: DatabaseLike,
    team_id: int,
) -> Sequence[int]:
    """Return the members belonging to the team or to one of its descendant teams."""
    team_rows = await fetch_teams_recursively(account, sdb, [Team.members], [team_id])
    members = set(chain.from_iterable(row[Team.members.name] for row in team_rows))
    return list(members)


@sentry_span
async def delete_team(team: Row, sdb_conn: Connection) -> None:
    """Delete a Team row from the DB.

    Related objects like TeamGoals are also deleted.
    """
    assert await conn_in_transaction(sdb_conn)
    if (parent_id := team[Team.parent_id.name]) is None:
        raise ResponseError(BadRequestError(detail="Root team cannot be deleted."))
    team_id = team[Team.id.name]

    await sdb_conn.execute(
        sa.update(Team)
        .where(Team.parent_id == team_id)
        .values({Team.parent_id: parent_id, Team.updated_at: datetime.now(timezone.utc)}),
    )
    await sdb_conn.execute(sa.delete(Team).where(Team.id == team_id))
    # TeamGoal-s have been deleted by ON DELETE CASCADE on team_id, now empty Goals must be removed
    await delete_empty_goals(team[Team.owner_id.name], sdb_conn)


@sentry_span
async def sync_team_members(
    team: Row,
    members: Sequence[int],
    sdb_conn: Connection,
) -> None:
    """Update the members of the Team.

    No update is done if team already contains exactly the same members.

    Do not return anything because we can either delete or insert people.
    """
    assert await conn_in_transaction(sdb_conn)

    if (members := sorted(members)) != team[Team.members.name]:
        # invariant: the team members are always sorted, so there is no need to compare sets
        await sdb_conn.execute(
            sa.update(Team)
            .where(Team.id == team[Team.id.name])
            .values({Team.updated_at: datetime.now(timezone.utc), Team.members: members}),
        )
