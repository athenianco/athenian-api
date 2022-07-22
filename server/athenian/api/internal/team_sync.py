from __future__ import annotations

from datetime import datetime, timezone
import graphlib
import re
from typing import Any, Sequence

import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.align.goals.dbaccess import delete_empty_goals
from athenian.api.db import Connection, Database, DatabaseLike, Row, is_postgresql
from athenian.api.internal.team import (
    get_meta_teams_members,
    get_meta_teams_topological_order,
    get_root_team,
)
from athenian.api.models.metadata.github import (
    Organization as MetadataOrganization,
    Team as MetadataTeam,
)
from athenian.api.models.state.models import Team


async def sync_teams(
    account: int,
    meta_ids: Sequence[int],
    sdb: Database,
    mdb: DatabaseLike,
) -> None:
    """Sync the teams from metadata DB to state DB."""
    meta_teams = await _MetaTeams.from_db(meta_ids, mdb)
    all_members = await meta_teams.get_all_members()
    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            state_teams = await _StateTeams.from_db(account, sdb_conn)

            new_meta_teams = meta_teams.get_new(state_teams)
            existing_meta_teams = meta_teams.get_existing(state_teams)
            gone_state_teams = state_teams.get_gone(meta_teams)

            db_ops = _StateDBOperator(account, sdb_conn)
            await db_ops.insert_new(new_meta_teams, all_members, state_teams)
            await db_ops.update_existing(existing_meta_teams, all_members, state_teams)
            await db_ops.delete(gone_state_teams)
            await db_ops.revert_temporary_names()


class SyncTeamsError(Exception):
    """An error occurred during teams sync operation."""


class _MetaTeams:
    def __init__(self, rows: Sequence[Row], meta_ids: Sequence[int], mdb: DatabaseLike):
        self._rows = rows
        self._meta_ids = meta_ids
        self._mdb = mdb
        self._by_id = {r[MetadataTeam.id.name]: r for r in rows}

    @classmethod
    async def from_db(cls, meta_ids: Sequence[int], mdb: DatabaseLike) -> _MetaTeams:
        meta_orgs_query = sa.select(MetadataOrganization.id).where(
            MetadataOrganization.acc_id.in_(meta_ids),
        )
        columns = (MetadataTeam.name, MetadataTeam.id, MetadataTeam.parent_team_id)
        query = sa.select(columns).where(
            MetadataTeam.acc_id.in_(meta_ids), MetadataTeam.organization_id.in_(meta_orgs_query),
        )
        return cls(await mdb.fetch_all(query), meta_ids, mdb)

    def get_new(self, state_teams: _StateTeams) -> Sequence[Row]:
        existing = state_teams.existing_by_origin_node_id
        return [r for r in self._rows if r[MetadataTeam.id.name] not in existing]

    def get_existing(self, state_teams: _StateTeams) -> Sequence[Row]:
        existing = state_teams.existing_by_origin_node_id
        return [r for r in self._rows if r[MetadataTeam.id.name] in existing]

    async def get_all_members(self) -> dict[int, list[int]]:
        ids = [r[MetadataTeam.id.name] for r in self._rows]
        return await get_meta_teams_members(ids, self._meta_ids, self._mdb)

    @property
    def by_id(self) -> dict[int, Row]:
        return self._by_id


class _StateTeams:
    def __init__(self, rows: Sequence[Row], root_team_id: int):
        self._rows = rows
        self._root_team_id = root_team_id
        self._existing = {id_: r for r in rows if (id_ := r[Team.origin_node_id.name]) is not None}
        self._created: dict[int, int] = {}

    @classmethod
    async def from_db(cls, account: int, sdb_conn: Connection) -> _StateTeams:
        where = [Team.owner_id == account, Team.name != Team.BOTS, Team.parent_id.isnot(None)]
        query = sa.select(Team).where(*where)
        root_team_id = (await get_root_team(account, sdb_conn))[Team.id.name]
        rows = await sdb_conn.fetch_all(query)
        return cls(rows, root_team_id)

    def check_unmapped(self) -> None:
        unmapped = [r[Team.id.name] for r in self._rows if r[Team.origin_node_id.name] is None]
        if unmapped:
            raise SyncTeamsError(
                f"Cannot sync teams, account has some teams not mapped to mdb: {unmapped}",
            )

    @property
    def existing_by_origin_node_id(self) -> dict[int, Row]:
        return self._existing

    @property
    def root_team_id(self) -> int:
        return self._root_team_id

    def track_created_team(self, id_: int, origin_node_id) -> None:
        self._created[origin_node_id] = id_

    def get_id(self, origin_node_id: int) -> int:
        existing_row = self._existing.get(origin_node_id)
        if existing_row is not None:
            return existing_row[Team.id.name]
        return self._created[origin_node_id]

    def get_gone(self, meta_teams) -> Sequence[Row]:
        rows = self._existing.values()
        return [r for r in rows if r[Team.origin_node_id.name] not in meta_teams.by_id]


class _NameMangler:
    _TMP_PREFIX = "$32$àē_ì_ōū$32$"

    @classmethod
    def apply(cls, name: str) -> str:
        return f"{cls._TMP_PREFIX}{name}"

    @classmethod
    def applied_where_clause(cls, column):
        return Team.name.like(f"{cls._TMP_PREFIX}%")

    @classmethod
    async def revert_sql_expr(cls, column, db: DatabaseLike):
        if await is_postgresql(db):
            prefix_re_escaped = re.escape(cls._TMP_PREFIX)
            return sa.func.regexp_replace(column, f"^{prefix_re_escaped}", "")
        else:
            return sa.func.replace(column, cls._TMP_PREFIX, "")


class _StateDBOperator:
    def __init__(self, account: int, sdb_conn: Connection):
        self._account = account
        self._sdb_conn = sdb_conn

    async def insert_new(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> None:
        for meta_team_row in self._new_meta_teams_insertion_order(meta_team_rows):
            origin_node_id = meta_team_row[MetadataTeam.id.name]
            members = sorted(all_members.get(origin_node_id, []))
            name = _NameMangler.apply(meta_team_row[MetadataTeam.name.name])

            if (meta_parent_id := meta_team_row[MetadataTeam.parent_team_id.name]) is None:
                parent_id = state_teams.root_team_id
            else:
                # parent team can be either created in a *previous* iteration
                # (thanks to _new_meta_teams_insertion_order) or in existing teams
                parent_id = state_teams.get_id(meta_parent_id)

            team = Team(
                owner_id=self._account,
                name=name,
                members=members,
                parent_id=parent_id,
                origin_node_id=origin_node_id,
            )
            values = team.create_defaults().explode()
            new_team_id = await self._sdb_conn.execute(sa.insert(Team).values(values))
            state_teams.track_created_team(new_team_id, origin_node_id)

    async def update_existing(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> None:
        stmts = []
        for meta_team_row in meta_team_rows:
            members = all_members[meta_team_row[MetadataTeam.id.name]]
            updates = _get_team_updates(meta_team_row, members, state_teams)
            if updates:
                updates[Team.updated_at.name] = datetime.now(timezone.utc)
                team_id = state_teams.get_id(meta_team_row[MetadataTeam.id.name])
                stmts.append(sa.update(Team).where(Team.id == team_id).values(updates))

        for stmt in stmts:
            await self._sdb_conn.execute(stmt)

    async def delete(self, state_team_rows: Sequence[Row]) -> None:
        team_ids = [r[Team.id.name] for r in state_team_rows]
        if team_ids:
            await self._sdb_conn.execute(sa.delete(Team).where(Team.id.in_(team_ids)))
            await delete_empty_goals(self._account, self._sdb_conn)

    async def revert_temporary_names(self) -> None:
        value = await _NameMangler.revert_sql_expr(Team.name, self._sdb_conn)
        query = (
            sa.update(Team)
            .where(_NameMangler.applied_where_clause(Team.name))
            .values({Team.name: value, Team.updated_at: datetime.now(timezone.utc)})
        )
        await self._sdb_conn.execute(query)

    @classmethod
    def _new_meta_teams_insertion_order(cls, new_meta_teams: Sequence[Row]) -> Sequence[Row]:
        try:
            order = get_meta_teams_topological_order(new_meta_teams)
        except graphlib.CycleError:
            raise SyncTeamsError("Parent relation in metadata teams contains cycles")
        order_by_id = {team_id: idx for idx, team_id in enumerate(order)}
        return sorted(new_meta_teams, key=lambda r: order_by_id[r[MetadataTeam.id.name]])


def _get_team_updates(meta_row: Row, members: list[int], state_teams: _StateTeams) -> dict:
    updates: dict[InstrumentedAttribute, Any] = {}
    state_team_row = state_teams.existing_by_origin_node_id[meta_row[MetadataTeam.id.name]]
    if state_team_row[Team.name.name] != (meta_name := meta_row[MetadataTeam.name.name]):
        updates[Team.name] = _NameMangler.apply(meta_name)

    if sorted(state_team_row[Team.members.name]) != members:
        updates[Team.members.name] = members

    meta_parent_id = meta_row[MetadataTeam.parent_team_id.name]
    if meta_parent_id is None:
        parent_id = state_teams.root_team_id
    else:
        parent_id = state_teams.get_id(meta_parent_id)
    if state_team_row[Team.parent_id.name] != parent_id:
        updates[Team.parent_id.name] = parent_id

    return updates
