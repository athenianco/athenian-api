from aiohttp import web
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_account_repositories, get_metadata_account_ids
from athenian.api.controllers.jira import load_mapped_jira_users
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import Settings
from athenian.api.models.metadata.github import User
from athenian.api.models.state.models import UserAccount
from athenian.api.models.web import Contributor, NotFoundError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@weight(0.5)
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
        tasks = [
            get_account_repositories(id, True, sdb_conn),
            #                            not sdb_conn! we must go parallel
            get_metadata_account_ids(id, request.sdb, request.cache),
        ]
        repos, meta_ids = await gather(*tasks)
        prefixer = Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
        release_settings = \
            await Settings.from_request(request, account_id).list_release_matches(repos)
        repos = [r.split("/", 1)[1] for r in repos]
        users = await mine_contributors(
            repos, None, None, False, [], release_settings,
            account_id, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
        mapped_jira = await load_mapped_jira_users(
            account_id, [u[User.node_id.key] for u in users], sdb_conn, request.mdb, request.cache)
        prefixer = await prefixer.load()
        contributors = [
            Contributor(login=prefixer.user_node_map[u[User.node_id.key]],
                        name=u[User.name.key],
                        email="<classified>",  # u[User.email.key] TODO(vmarkovtsev): DEV-87
                        picture=u[User.avatar_url.key],
                        jira_user=mapped_jira.get(u[User.node_id.key]))
            for u in users
        ]
        return model_response(contributors)
