from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime, timezone
import graphlib
import logging
import re
from typing import Any, Iterator, Optional, Sequence

import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.align.goals.dbaccess import delete_empty_goals
from athenian.api.db import Connection, Database, DatabaseLike, Row, is_postgresql
from athenian.api.internal.team import get_root_team
from athenian.api.internal.team_meta import (
    get_meta_teams_members,
    get_meta_teams_topological_order,
)
from athenian.api.models.metadata.github import (
    Organization as MetadataOrganization,
    Team as MetadataTeam,
)
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.models.state.models import Team

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TeamSyncMetrics:
    """Team synchronization statistics."""

    added: int
    removed: int
    updated: int

    @classmethod
    def empty(cls) -> TeamSyncMetrics:
        """Initialize a new TeamSyncMetrics instance filled with zeros."""
        return TeamSyncMetrics(0, 0, 0)

    def as_db(self) -> Iterator[HealthMetric]:
        """Generate HealthMetric-s from this instance."""
        yield HealthMetric(name="added_teams", value=self.added)
        yield HealthMetric(name="removed_teams", value=self.removed)
        yield HealthMetric(name="updated_teams", value=self.updated)


async def sync_teams(
    account: int,
    meta_ids: Sequence[int],
    sdb: Database,
    mdb: DatabaseLike,
    *,
    dry_run: bool = False,
    force: bool = False,
    unmapped: bool = False,
    metrics: Optional[TeamSyncMetrics] = None,
) -> None:
    """
    Sync the teams from metadata DB to state DB.

    :param force: Value indicating whether to delete all the teams which are mismatched in mdb. \
                  If False, we abort the call in this situation.
    :param unmapped: Value indicating whether we shall delete any team without `origin_node_id`.
    :param metrics: Optional health metrics to populate.
    """
    meta_teams = await _MetaTeams.from_db(meta_ids, mdb)
    all_members = await meta_teams.get_all_members()
    log.info(
        "executing team sync for account %s%s", account, " in dry run mode" if dry_run else "",
    )
    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            state_teams = await _StateTeams.from_db(account, sdb_conn, force, unmapped)

            new_meta_teams, existing_meta_teams = meta_teams.triage_new_and_existing(state_teams)
            gone_state_teams = state_teams.get_gone(meta_teams)

            db_ops = _DryRunStateDBOperator() if dry_run else _StateDBOperator(account, sdb_conn)
            await db_ops.insert_new(new_meta_teams, all_members, state_teams)
            updates_count = await db_ops.update_existing(
                existing_meta_teams, all_members, state_teams,
            )
            await db_ops.delete(gone_state_teams)
            await db_ops.revert_temporary_names()

            if metrics is not None:
                metrics.added = len(new_meta_teams)
                metrics.removed = len(gone_state_teams)
                metrics.updated = updates_count


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

    def triage_new_and_existing(self, state_teams: _StateTeams) -> tuple[list[Row], list[Row]]:
        existing_state_teams = state_teams.existing_by_origin_node_id
        new: list[Row] = []
        existing: list[Row] = []
        for r in self._rows:
            target = existing if r[MetadataTeam.id.name] in existing_state_teams else new
            target.append(r)
        return new, existing

    async def get_all_members(self) -> dict[int, list[int]]:
        ids = [r[MetadataTeam.id.name] for r in self._rows]
        return await get_meta_teams_members(ids, self._meta_ids, self._mdb)

    @property
    def by_id(self) -> dict[int, Row]:
        return self._by_id


class _StateTeams:
    def __init__(self, rows: Sequence[Row], root_team_id: int, force: bool, unmapped: bool):
        self._rows = rows
        self._root_team_id = root_team_id
        self._existing = {
            id_: r
            for r in rows
            if (id_ := r[Team.origin_node_id.name]) is not None
            or (unmapped and id_ != root_team_id and r[Team.name.name] != Team.BOTS)
        }
        self._created: dict[int, int] = {}
        self.force = force
        self.unmapped = unmapped
        self._check_unmapped()

    @classmethod
    async def from_db(
        cls,
        account: int,
        sdb_conn: Connection,
        force: bool,
        unmapped: bool,
    ) -> _StateTeams:
        where = [Team.owner_id == account, Team.name != Team.BOTS, Team.parent_id.isnot(None)]
        query = sa.select(Team).where(*where)
        root_team_id = (await get_root_team(account, sdb_conn))[Team.id.name]
        rows = await sdb_conn.fetch_all(query)
        return cls(rows, root_team_id, force, unmapped)

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

    def get_parent_id(self, parent_origin_node_id: Optional[int]) -> int:
        if parent_origin_node_id is None:
            return self._root_team_id
        try:
            return self.get_id(parent_origin_node_id)
        except KeyError:  # this happens when meta team has an invalid, inexisting parent
            return self._root_team_id

    def get_gone(self, meta_teams) -> Sequence[Row]:
        rows = self._existing.values()
        return [r for r in rows if r[Team.origin_node_id.name] not in meta_teams.by_id]

    def _check_unmapped(self) -> None:
        if self.force:
            return
        unmapped = [r[Team.id.name] for r in self._rows if r[Team.origin_node_id.name] is None]
        if unmapped:
            raise SyncTeamsError(
                f"Cannot sync teams, account has some teams not mapped to mdb: {unmapped}",
            )


class _NameMangler:
    # github team names cannot include a "$"
    _TMP_PREFIX = "$"

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


class _StateDBOperatorI(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def insert_new(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> None:
        ...

    @abc.abstractmethod
    async def update_existing(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> int:
        ...

    @abc.abstractmethod
    async def delete(self, state_team_rows: Sequence[Row]) -> None:
        ...

    @abc.abstractmethod
    async def revert_temporary_names(self) -> None:
        ...


def _ensure_name_no_clash_with_builtin(name: str) -> str:
    if name == Team.BOTS or name == Team.ROOT:
        name += ".github"
    return name


class _StateDBOperator(_StateDBOperatorI):
    def __init__(self, account: int, sdb_conn: Connection):
        self._account = account
        self._sdb_conn = sdb_conn

    async def insert_new(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> None:
        for meta_team_row in _new_meta_teams_insertion_order(meta_team_rows):
            origin_node_id = meta_team_row[MetadataTeam.id.name]
            members = sorted(all_members.get(origin_node_id, []))
            real_name = _ensure_name_no_clash_with_builtin(meta_team_row[MetadataTeam.name.name])
            name = _NameMangler.apply(real_name)

            # parent team can be either created in a *previous* iteration
            # (thanks to _new_meta_teams_insertion_order) or in existing teams
            parent_id = state_teams.get_parent_id(meta_team_row[MetadataTeam.parent_team_id.name])

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
            log.info('created new team "%s" with origin_node_id %s', real_name, origin_node_id)

    async def update_existing(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> int:
        stmts = []
        updates_count = 0
        for meta_team_row in meta_team_rows:
            members = all_members[meta_team_row[MetadataTeam.id.name]]
            updates = _get_team_updates(meta_team_row, members, state_teams)
            if updates:
                updates_count += 1
                team_id = state_teams.get_id(meta_team_row[MetadataTeam.id.name])
                updated_fields = [col.name for col in updates]
                log.info("updating fields %s for team %s", ",".join(updated_fields), team_id)
                updates[Team.updated_at.name] = datetime.now(timezone.utc)
                stmts.append(sa.update(Team).where(Team.id == team_id).values(updates))

        for stmt in stmts:
            await self._sdb_conn.execute(stmt)
        return updates_count

    async def delete(self, state_team_rows: Sequence[Row]) -> None:
        if team_ids := [r[Team.id.name] for r in state_team_rows]:
            await self._sdb_conn.execute(sa.delete(Team).where(Team.id.in_(team_ids)))
            await delete_empty_goals(self._account, self._sdb_conn)
            log.info("deleted teams: %s", ", ".join(map(str, team_ids)))

    async def revert_temporary_names(self) -> None:
        value = await _NameMangler.revert_sql_expr(Team.name, self._sdb_conn)
        query = (
            sa.update(Team)
            .where(_NameMangler.applied_where_clause(Team.name))
            .values({Team.name: value, Team.updated_at: datetime.now(timezone.utc)})
        )
        await self._sdb_conn.execute(query)


class _DryRunStateDBOperator(_StateDBOperatorI):
    async def insert_new(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> None:
        for meta_team_row in _new_meta_teams_insertion_order(meta_team_rows):
            origin_node_id = meta_team_row[MetadataTeam.id.name]
            name = _ensure_name_no_clash_with_builtin(meta_team_row[MetadataTeam.name.name])
            log.info('creating team "%s" with origin_node_id %s', name, origin_node_id)
            state_teams.track_created_team(-1, origin_node_id)

    async def update_existing(
        self,
        meta_team_rows: Sequence[Row],
        all_members: dict[int, list[int]],
        state_teams: _StateTeams,
    ) -> int:
        updates_count = 0
        for meta_team_row in meta_team_rows:
            members = all_members[meta_team_row[MetadataTeam.id.name]]
            if updates := _get_team_updates(meta_team_row, members, state_teams):
                updates_count += 1
                team_id = state_teams.get_id(meta_team_row[MetadataTeam.id.name])
                updated_fields = [col.name for col in updates]
                log.info("updating fields %s for team %s", ",".join(updated_fields), team_id)
        return updates_count

    async def delete(self, state_team_rows: Sequence[Row]) -> None:
        team_ids = [r[Team.id.name] for r in state_team_rows]
        if team_ids:
            log.info("teams would be deleted: %s", ", ".join(map(str, team_ids)))

    async def revert_temporary_names(self) -> None:
        ...


def _get_team_updates(meta_row: Row, members: list[int], state_teams: _StateTeams) -> dict:
    updates: dict[InstrumentedAttribute, Any] = {}
    state_team_row = state_teams.existing_by_origin_node_id[meta_row[MetadataTeam.id.name]]
    if state_team_row[Team.name.name] != (
        meta_name := _ensure_name_no_clash_with_builtin(meta_row[MetadataTeam.name.name])
    ):
        updates[Team.name] = _NameMangler.apply(meta_name)

    if state_team_row[Team.members.name] != (sorted_members := sorted(members)):
        updates[Team.members] = sorted_members

    parent_id = state_teams.get_parent_id(meta_row[MetadataTeam.parent_team_id.name])
    if state_team_row[Team.parent_id.name] != parent_id:
        updates[Team.parent_id] = parent_id

    return updates


def _new_meta_teams_insertion_order(new_meta_teams: Sequence[Row]) -> Sequence[Row]:
    try:
        order = get_meta_teams_topological_order(new_meta_teams)
    except graphlib.CycleError:
        raise SyncTeamsError("Parent relation in metadata teams contains cycles")
    order_by_id = {team_id: idx for idx, team_id in enumerate(order)}
    return sorted(new_meta_teams, key=lambda r: order_by_id[r[MetadataTeam.id.name]])
