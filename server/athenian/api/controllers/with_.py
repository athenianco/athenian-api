from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Set, Union

from morcilla import Database
import numpy as np
from sqlalchemy import and_, select

from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.user import MANNEQUIN_PREFIX
from athenian.api.models.state.models import Team
from athenian.api.models.web import InvalidRequestError, NotFoundError
from athenian.api.response import ResponseError


async def resolve_withgroups(model_withgroups: Optional[Iterable[Any]],
                             kind: Any,
                             dereference: bool,
                             account: int,
                             prefix: Optional[str],
                             position: str,
                             prefixer: Prefixer,
                             sdb: Database,
                             group_type: callable = lambda i: i,
                             ) -> List[Dict[Any, Collection[int]]]:
    """
    Load IDs or normalize logins of one or more groups of participants.

    :param model_withgroups: Several "with" maps.
    :param kind: Parse the participation key type to this enum.
    :param dereference: Value indicating whether to load user node IDs instead of the logins.
    :param prefix: Optional service prefix to validate.
    :param position: Path in the model to report good errors.
    :param group_type: Convert each group to this type.
    :return: len(model_withgroups) or less maps from participation kind to user node IDs or logins.
    """
    if model_withgroups is None:
        return []
    teams = set()
    for with_ in model_withgroups:
        for k, people in (with_ or {}).items():
            scan_for_teams(people, teams, f"{position}.{k}")
    teams_map = await fetch_teams_map(teams, account, sdb)
    result = []
    for with_ in model_withgroups:
        withgroup = {}
        for k, people in (with_ or {}).items():
            if not people:
                continue
            withgroup[kind[k.upper()]] = group_type(compile_developers(
                people, teams_map, prefix, dereference, prefixer, f"{position}.{k}"))
        if withgroup:
            result.append(withgroup)
    return result


def scan_for_teams(people: Optional[List[str]],
                   teams: Set[int],
                   pointer: str) -> None:
    """Extract and validate all the team mentions."""
    for j, person in enumerate(people or []):
        if not person.startswith("{"):
            continue
        try:
            if not person.endswith("}"):
                raise ValueError("the team format must be {<id>}")
            teams.add(int(person[1:-1]))
        except ValueError as e:
            raise ResponseError(InvalidRequestError(
                detail=str(e),
                pointer=f"{pointer}[{j}]",
            )) from e


async def fetch_teams_map(teams: Collection[int],
                          account: int,
                          sdb: Database,
                          ) -> Dict[int, List[int]]:
    """Load the mapping from team ID to member IDs."""
    if not teams:
        return {}
    rows = await sdb.fetch_all(
        select([Team.id, Team.members])
        .where(and_(
            Team.owner_id == account,
            Team.id.in_(teams),
        )))
    teams_map = {r[0]: r[1] for r in rows}
    if diff := (teams - teams_map.keys()):
        raise ResponseError(NotFoundError(
            detail=f"Some teams do not exist or access denied: {diff}",
        ))
    return teams_map


def compile_developers(developers: Optional[Iterable[str]],
                       teams: Dict[int, List[int]],
                       prefix: Optional[str],
                       dereference: bool,
                       prefixer: Prefixer,
                       pointer: str,
                       unique: bool = True,
                       ) -> Union[Sequence[str], Sequence[int]]:
    """
    Produce the final list of participants with resolved teams.

    :param dereference: Value indicating whether to return user node IDs instead of the logins.
    :param unique: Remove any duplicates.
    :return: If `unique`, a set, otherwise, a list.
    """
    devs = []
    prefix = prefix.rstrip("/") if prefix is not None else None
    user_node_to_login = prefixer.user_node_to_login.get
    user_login_to_nodes = prefixer.user_login_to_node.__getitem__
    for i, dev in enumerate(developers or []):
        if dev.startswith("{"):
            try:
                team = teams[int(dev[1:-1])]
            except KeyError:
                raise ResponseError(InvalidRequestError(
                    detail="Teams are not supported",
                    pointer=f"{pointer}[{i}]",
                ))
            if not dereference:
                team = {user_node_to_login(p) for p in team} - {None}
            devs.extend(team)
            continue
        parts = dev.split("/")
        dev_prefix, dev_login = parts[0], parts[-1]
        if prefix is not None and dev_prefix != prefix and dev_prefix != MANNEQUIN_PREFIX:
            raise ResponseError(InvalidRequestError(
                detail='User "%s" has prefix "%s" that does not match with the repository prefix '
                       '"%s"' % (dev, dev_prefix, prefix),
                pointer=f"{pointer}[{i}]",
            ))
        if dereference:
            try:
                devs.extend(user_login_to_nodes(dev_login))
            except KeyError as e:
                raise ResponseError(InvalidRequestError(
                    detail=f'User "{dev}" does not exist',
                    pointer=f"{pointer}[{i}]",
                )) from e
        else:
            devs.append(dev_login)
    if unique:
        return np.unique(devs)
    return devs
