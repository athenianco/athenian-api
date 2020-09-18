from aiohttp import web
from sqlalchemy import and_, select

from athenian.api.controllers.account import get_account_repositories
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.controllers.settings import Settings
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
        repos = await get_account_repositories(id, sdb_conn)
        release_settings = \
            await Settings.from_request(request, account_id).list_release_matches(repos)
        repos = [r.split("/", 1)[1] for r in repos]
        users = await mine_contributors(
            repos, None, None, False, [], release_settings,
            request.mdb, request.pdb, request.cache)
        prefix = PREFIXES["github"]
        contributors = [
            Contributor(login=f"{prefix}{u['login']}", name=u["name"],
                        email="<classified>",  # u["email"] TODO(vmarkovtsev): DEV-87
                        picture=u["avatar_url"])
            for u in users
        ]
        return model_response(contributors)
