from typing import List, Optional, Sequence, Tuple, Type, Union

import aiomcache
import databases
from sqlalchemy import select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.controllers.account import get_user_account_status
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import ForbiddenError, InvalidRequestError, NotFoundError
from athenian.api.response import ResponseError


async def resolve_reposet(repo: str,
                          pointer: str,
                          uid: str,
                          account: int,
                          db: Union[databases.core.Connection, databases.Database],
                          cache: Optional[aiomcache.Client],
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
    rs, _ = await fetch_reposet(set_id, [RepositorySet.items], uid, db, cache)
    if rs.owner != account:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to reference reposet %d in this query" %
                   (uid, set_id)))
    return rs.items


async def fetch_reposet(
    id: int,
    columns: Union[Sequence[Type[RepositorySet]], Sequence[InstrumentedAttribute]],
    uid: str,
    sdb: Union[databases.Database, databases.core.Connection],
    cache: Optional[aiomcache.Client],
) -> Tuple[RepositorySet, bool]:
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
    rs = await sdb.fetch_one(select(columns).where(RepositorySet.id == id))
    if rs is None or len(rs) == 0:
        raise ResponseError(NotFoundError(detail="Repository set %d does not exist" % id))
    account = rs[RepositorySet.owner.key]
    adm = await get_user_account_status(uid, account, sdb, cache)
    return RepositorySet(**rs), adm
