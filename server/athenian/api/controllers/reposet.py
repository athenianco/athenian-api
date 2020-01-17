from typing import List, Sequence, Type, Union, Optional, Tuple

import databases
from sqlalchemy import select, and_
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.auth import User
from athenian.api.controllers.response import ResponseError
from athenian.api.models.state.models import RepositorySet, UserTeam
from athenian.api.models.web import ForbiddenError, InvalidRequestError, NotFoundError


async def resolve_reposet(repo: str, pointer: str, db: databases.Database, user: User, team: int,
                          ) -> List[str]:
    """
    Dereference the repository sets.

    If `repo` is a regular repository, return `[repo]`. Otherwise, return the list of \
    repositories by the parsed ID from the database.
    """
    if not repo.startswith("{"):
        return [repo]
    if not repo.endswith("}"):
        raise ResponseError(InvalidRequestError(
            detail="repository set format is invalid: %s" % repo,
            pointer=pointer,
        ))
    try:
        set_id = int(repo[1:-1])
    except ValueError:
        raise ResponseError(InvalidRequestError(
            detail="repository set identifier is invalid: %s" % repo,
            pointer=pointer,
        ))
    rs, _ = await fetch_reposet(set_id, [RepositorySet.items], db, user, team)
    return rs.items


async def fetch_reposet(
        id: int, columns: Union[Sequence[Type[RepositorySet]], Sequence[InstrumentedAttribute]],
        db: databases.Database, user: User, team: Optional[int]) -> Tuple[RepositorySet, bool]:
    """
    Retrieve a repository set by ID and check the access for the given user.

    :return: Loaded RepositorySet and `is_admin` flag that indicates whether the user has \
             RW access to that set.
    """
    if not columns or columns[0] is not RepositorySet:
        for col in columns:
            if col is RepositorySet.owner:
                break
        else:
            columns = list(columns)
            columns.append(RepositorySet.owner)
    rs = await db.fetch_one(select(columns).where(RepositorySet.id == id))
    if rs is None or len(rs) == 0:
        raise ResponseError(NotFoundError(detail="Repository set %d does not exist" % id))
    if team is None:
        # if there is no team, suppose the reposet owner
        team = rs[RepositorySet.owner.key]
    elif rs[RepositorySet.owner.key] != team:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (user.id, id)))
    # finally check that the user belongs to the team
    status = await db.fetch_one(select([UserTeam.is_admin]).where(
        and_(UserTeam.user_id == user.id, UserTeam.team_id == team)))
    if status is None:
        raise ResponseError(ForbiddenError(
            detail="User %s does not belong to team %d" % (user.id, team)))
    return RepositorySet(**rs), status[UserTeam.is_admin.key]
