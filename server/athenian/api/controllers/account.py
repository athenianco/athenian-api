from collections import defaultdict
import logging
import pickle
from sqlite3 import IntegrityError, OperationalError
import struct
from typing import List, Optional, Tuple

import aiomcache
from asyncpg import UniqueViolationError
import networkx as nx
from sqlalchemy import and_, func, insert, join, select

from athenian.api import metadata
from athenian.api.cache import cached, max_exptime
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import NodeUser, Organization, Team as MetadataTeam, \
    TeamMember
from athenian.api.models.state.models import Account, AccountGitHubAccount, RepositorySet, \
    Team as StateTeam, UserAccount
from athenian.api.models.web import NoSourceDataError, NotFoundError
from athenian.api.response import ResponseError
from athenian.api.typing_utils import DatabaseLike


@cached(
    # the TTL is huge because this relation will never change and is requested frequently
    exptime=max_exptime,
    serialize=lambda ids: struct.pack("!" + "q" * len(ids), *ids),
    deserialize=lambda buf: struct.unpack("!" + "q" * (len(buf) // 8), buf),
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_metadata_account_ids(account: int,
                                   sdb: DatabaseLike,
                                   cache: Optional[aiomcache.Client],
                                   ) -> Tuple[int, ...]:
    """Fetch the metadata account IDs for the given API account ID."""
    ids = await sdb.fetch_all(select([AccountGitHubAccount.id])
                              .where(AccountGitHubAccount.account_id == account))
    if len(ids) == 0:
        acc_exists = await sdb.fetch_val(select([Account.id]).where(Account.id == account))
        if not acc_exists:
            raise ResponseError(NotFoundError(detail="Account %d does not exist" % account))
        raise ResponseError(NoSourceDataError(
            detail="The installation of account %d has not finished yet." % account))
    return tuple(r[0] for r in ids)


@cached(
    exptime=60,
    serialize=lambda is_admin: b"1" if is_admin else b"0",
    deserialize=lambda buf: buf == b"1",
    key=lambda user, account, **_: (user, account),
)
async def get_user_account_status(user: str,
                                  account: int,
                                  sdb: DatabaseLike,
                                  cache: Optional[aiomcache.Client],
                                  ) -> bool:
    """Return the value indicating whether the given user is an admin of the given account."""
    status = await sdb.fetch_val(
        select([UserAccount.is_admin])
        .where(and_(UserAccount.user_id == user, UserAccount.account_id == account)))
    if status is None:
        raise ResponseError(NotFoundError(
            detail="Account %d does not exist or user %s is not a member." % (account, user)))
    return status


async def get_account_repositories(account: int,
                                   sdb: DatabaseLike,
                                   with_prefix=True) -> List[str]:
    """Fetch all the repositories belonging to the account."""
    repos = await sdb.fetch_one(select([RepositorySet.items]).where(and_(
        RepositorySet.owner_id == account,
        RepositorySet.name == RepositorySet.ALL,
    )))
    if repos is None:
        raise ResponseError(NoSourceDataError(
            detail="The installation of account %d has not finished yet." % account))
    repos = repos[0]
    if not with_prefix:
        repos = [r.split("/", 1)[1] for r in repos]
    return repos


@cached(
    exptime=24 * 60 * 60,  # 1 day
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda account, **_: (account,),
)
async def get_account_organizations(account: int,
                                    sdb: DatabaseLike,
                                    mdb: DatabaseLike,
                                    cache: Optional[aiomcache.Client],
                                    ) -> List[Organization]:
    """Fetch the list of GitHub organizations installed for the account."""
    ghids = await get_metadata_account_ids(account, sdb, cache)
    rows = await mdb.fetch_all(select([Organization]).where(Organization.acc_id.in_(ghids)))
    return [Organization(**r) for r in rows]


async def copy_teams_as_needed(account: int,
                               meta_ids: Tuple[int, ...],
                               sdb: DatabaseLike,
                               mdb: DatabaseLike,
                               cache: Optional[aiomcache.Client],
                               ) -> List[str]:
    """
    Copy the teams from GitHub organization if none exist yet.

    :return: List of created team names.
    """
    log = logging.getLogger("%s.create_teams_as_needed" % metadata.__package__)
    existing = await sdb.fetch_val(select([func.count(StateTeam.id)])
                                   .where(StateTeam.owner_id == account))
    if existing > 0:
        log.info("Found %d existing teams for account %d, no-op", existing, account)
    orgs = [org.id for org in await get_account_organizations(account, sdb, mdb, cache)]
    team_rows = await mdb.fetch_all(select([MetadataTeam])
                                    .where(and_(MetadataTeam.organization.in_(orgs),
                                                MetadataTeam.acc_id.in_(meta_ids))))
    if not team_rows:
        log.warning("Found 0 metadata teams for account %d", account)
        return []
    # check for cycles - who knows?
    dig = nx.DiGraph()
    for row in team_rows:
        if (parent_id := row[MetadataTeam.parent_team.key]) is not None:
            dig.add_edge(row[MetadataTeam.id.key], parent_id)
    try:
        cycle = nx.find_cycle(dig)
    except nx.NetworkXNoCycle:
        pass
    else:
        log.error("Found a metadata parent-child team reference cycle: %s", cycle)
        return []
    teams = {t[MetadataTeam.id.key]: t for t in team_rows}
    member_rows = await mdb.fetch_all(
        select([TeamMember.parent_id, NodeUser.login]).select_from(
            join(TeamMember, NodeUser, and_(TeamMember.child_id == NodeUser.id,
                                            TeamMember.acc_id == NodeUser.acc_id)),
        ).where(and_(TeamMember.parent_id.in_(teams), TeamMember.acc_id.in_(meta_ids))))
    members = defaultdict(list)
    prefix = PREFIXES["github"]
    for row in member_rows:
        members[row[TeamMember.parent_id.key]].append(prefix + row[NodeUser.login.key])
    db_ids = {}
    for node_id in reversed(list(nx.topological_sort(dig))):
        team = teams[node_id]
        if (parent := teams.get(team[MetadataTeam.parent_team.key])) is not None:
            parent = db_ids[parent[MetadataTeam.id.key]]
        team = StateTeam(owner_id=account,
                         name=team[MetadataTeam.name.key],
                         members=sorted(members.get(team[MetadataTeam.id.key], [])),
                         parent_id=parent,
                         ).create_defaults().explode()
        try:
            db_ids[node_id] = await sdb.execute(insert(StateTeam).values(team))
        except (UniqueViolationError, IntegrityError, OperationalError) as e:
            log.error('Failed to create team "%s" in account %d: %s',
                      team[StateTeam.name.key], account, e)
            db_ids[node_id] = None
    team_names = [t[MetadataTeam.name.key] for t in team_rows]
    log.info("Created %d teams in account %d: %s", len(team_names), account, team_names)
    return team_names
