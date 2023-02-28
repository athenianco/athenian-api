from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Collection, Iterable, KeysView, Mapping, Optional, Sequence

import aiomcache
import numpy as np

from athenian.api.db import Database, DatabaseLike
from athenian.api.internal.jira import load_mapped_jira_users
from athenian.api.internal.miners.participation import JIRAParticipants
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.user import MANNEQUIN_PREFIX
from athenian.api.models.state.models import Team
from athenian.api.models.web import InvalidRequestError, JIRAFilterWith, NotFoundError
from athenian.api.response import ResponseError


async def resolve_withgroups(
    model_withgroups: Optional[Iterable[Any]],
    kind: Any,
    dereference: bool,
    account: int,
    prefix: Optional[str],
    position: str,
    prefixer: Prefixer,
    sdb: DatabaseLike,
    group_type: Callable = lambda i: i,
) -> list[dict[Any, Collection[int]]]:
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
    teams: set[int] = set()
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
            withgroup[kind[k.upper()]] = group_type(
                compile_developers(
                    people, teams_map, prefix, dereference, prefixer, f"{position}.{k}",
                ),
            )
        if withgroup:
            result.append(withgroup)
    return result


def scan_for_teams(people: Optional[list[str]], teams: set[int], pointer: str) -> None:
    """Extract and validate all the team mentions."""
    for j, person in enumerate(people or []):
        if not person.startswith("{"):
            continue
        try:
            if not person.endswith("}"):
                raise ValueError("the team format must be {<id>}")
            teams.add(int(person[1:-1]))
        except ValueError as e:
            raise ResponseError(
                InvalidRequestError(
                    detail=str(e),
                    pointer=f"{pointer}[{j}]",
                ),
            ) from e


async def fetch_teams_map(
    teams: Collection[int],
    account: int,
    sdb: DatabaseLike,
) -> dict[int, list[int]]:
    """Load the mapping from team ID to member IDs."""
    if not teams:
        return {}

    team_rows = await fetch_teams_recursively(
        account, sdb, select_entities=(Team.id, Team.members, Team.parent_id), root_team_ids=teams,
    )
    teams_map = flatten_teams(team_rows)
    if not isinstance(teams, (set, KeysView)):
        teams = set(teams)
    if diff := (teams - teams_map.keys()):
        raise ResponseError(
            NotFoundError(detail=f"Some teams do not exist or access denied: {diff}"),
        )
    return teams_map


def flatten_teams(team_rows: Sequence[Mapping[int | str, Any]]) -> dict[int, list[int]]:
    """Union all the child teams with each root team.

    `team_rows` must be breadth first sorted.
    """
    teams_map: dict[int, set[int]] = defaultdict(set)
    # iter children before parents
    for row in reversed(team_rows):
        members, parent_id, team_id = (
            row[Team.members.name],
            row[Team.parent_id.name],
            row[Team.id.name],
        )

        teams_map[team_id] = teams_map[team_id].union(members)
        if parent_id is not None:
            teams_map[parent_id] |= teams_map[team_id]

    return {team_id: sorted(members) for team_id, members in teams_map.items()}


def compile_developers(
    developers: Optional[Iterable[str]],
    teams: dict[int, list[int]],
    prefix: Optional[str],
    dereference: bool,
    prefixer: Prefixer,
    pointer: str,
    unique: bool = True,
) -> Sequence[str] | Sequence[int]:
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
                raise ResponseError(
                    InvalidRequestError(
                        detail="Teams are not supported",
                        pointer=f"{pointer}[{i}]",
                    ),
                )
            if not dereference:
                team = {user_node_to_login(p) for p in team} - {None}
            devs.extend(team)
            continue
        parts = dev.split("/")
        dev_prefix, dev_login = parts[0], parts[-1]
        if prefix is not None and dev_prefix != prefix and dev_prefix != MANNEQUIN_PREFIX:
            raise ResponseError(
                InvalidRequestError(
                    detail=(
                        'User "%s" has prefix "%s" that does not match with the repository prefix '
                        '"%s"'
                    )
                    % (dev, dev_prefix, prefix),
                    pointer=f"{pointer}[{i}]",
                ),
            )
        if dereference:
            try:
                devs.extend(user_login_to_nodes(dev_login))
            except KeyError as e:
                raise ResponseError(
                    InvalidRequestError(
                        detail=f'User "{dev}" does not exist',
                        pointer=f"{pointer}[{i}]",
                    ),
                ) from e
        else:
            devs.append(dev_login)
    if unique:
        return np.unique(devs)
    return devs


async def resolve_jira_with(
    with_: list[JIRAFilterWith] | None,
    account: int,
    sdb: Database,
    mdb: Database,
    cache: aiomcache.Client | None,
) -> list[JIRAParticipants]:
    """Resolve the `JIRAFilterWith` received from outside in JIRAParticipants objects.

    Teams are dereferenced.

    """
    if not with_:
        return []
    teams = set()
    for i, group in enumerate(with_):
        for topic in ("assignees", "reporters", "commenters"):
            for j, dev in enumerate(getattr(group, topic, []) or []):
                if dev is not None and dev.startswith("{"):
                    try:
                        if not dev.endswith("}"):
                            raise ValueError
                        teams.add(int(dev[1:-1]))
                    except ValueError:
                        raise ResponseError(
                            InvalidRequestError(
                                pointer=f".with[{i}].{topic}[{j}]",
                                detail=f"Invalid team ID: {dev}",
                            ),
                        )
    teams_map = await fetch_teams_map(teams, account, sdb)
    all_team_members = set(chain.from_iterable(teams_map.values()))
    jira_map = await load_mapped_jira_users(account, all_team_members, sdb, mdb, cache)
    del all_team_members
    deref = []
    for group in with_:
        new_group = {}
        changed = False
        for topic in ("assignees", "reporters", "commenters"):
            if topic_devs := getattr(group, topic):
                new_topic_devs = []
                topic_changed = False
                for dev in topic_devs:
                    if dev is not None and dev.startswith("{"):
                        topic_changed = True
                        for member in teams_map[int(dev[1:-1])]:
                            try:
                                new_topic_devs.append(jira_map[member])
                            except KeyError:
                                continue
                    else:
                        new_topic_devs.append(dev)
                if topic_changed:
                    changed = True
                    new_group[topic] = new_topic_devs
                else:
                    new_group[topic] = topic_devs
        deref.append(JIRAFilterWith(**new_group) if changed else group)
    return [w.as_participants() for w in deref]
    return deref
