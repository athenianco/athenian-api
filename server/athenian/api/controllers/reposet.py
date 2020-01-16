from typing import List, Sequence, Type, Union

import databases
from sqlalchemy import select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.auth import User
from athenian.api.controllers.response import ResponseError
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import ForbiddenError, InvalidRequestError, NotFoundError


async def resolve_reposet(repo: str, pointer: str, db: databases.Database, user: User,
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
    rs = await fetch_reposet(set_id, [RepositorySet.items], db, user)
    return rs.items


async def fetch_reposet(
        id: int, columns: Union[Sequence[Type[RepositorySet]], Sequence[InstrumentedAttribute]],
        db: databases.Database, user: User) -> RepositorySet:
    """Retrieve a repository set by ID and check the access for the given user."""
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
    if rs[RepositorySet.owner.key] != user.team_id:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (user.id, id)))
    return RepositorySet(**rs)
