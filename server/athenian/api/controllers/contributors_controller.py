from aiohttp import web
from sqlalchemy import and_, select

from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.state.models import UserAccount
from athenian.api.models.web import Contributor, NotFoundError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


async def get_contributors(request: AthenianWebRequest, id: int) -> web.Response:
    """List all the contributors belonging to the specified account.

    :param id: Numeric identifier of the account.
    """
    async with request.sdb.connection() as sdb_conn:
        account_id = await sdb_conn.fetch_val(
            select([UserAccount.account_id])
            .where(and_(UserAccount.user_id == request.uid, UserAccount.account_id == id)))
        if account_id is None:
            err_detail = (
                f"Account {account_id} does not exist or user {request.uid} "
                "is not a member."
            )
            return ResponseError(NotFoundError(detail=err_detail)).response

        try:
            checker = await access_classes["github"](
                account_id, sdb_conn, request.mdb, request.cache).load()
        except ResponseError as e:
            return e.response

        repos = checker.installed_repos(with_prefix=False)
        users = await mine_contributors(repos, request.mdb, with_stats=False,
                                        cache=request.cache)
        prefix = PREFIXES["github"]
        contributors = [
            Contributor(login=f"{prefix}{u['login']}", name=u["name"],
                        email=u["email"], picture=u["avatar_url"])
            for u in users
        ]
        return model_response(contributors)
