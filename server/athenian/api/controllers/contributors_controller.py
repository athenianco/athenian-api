from aiohttp import web
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.cache import expires_header, short_term_exptime
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


@expires_header(short_term_exptime)
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
            raise ResponseError(NotFoundError(detail=err_detail))
        tasks = [
            get_account_repositories(id, True, sdb_conn),
            #                            not sdb_conn! we must go parallel
            get_metadata_account_ids(id, request.sdb, request.cache),
        ]
        repos, meta_ids = await gather(*tasks)
        prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
        settings = Settings.from_request(request, account_id)
        release_settings, logical_settings = await gather(
            settings.list_release_matches(repos),
            settings.list_logical_repositories(prefixer, repos),
        )
        repos = logical_settings.append_logical_repos([r.split("/", 1)[1] for r in repos])
        users = await mine_contributors(
            repos, None, None, False, [], release_settings, logical_settings, prefixer,
            account_id, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
        mapped_jira = await load_mapped_jira_users(
            account_id, [u[User.node_id.name] for u in users],
            sdb_conn, request.mdb, request.cache)
        contributors = [
            Contributor(login=prefixer.user_node_to_prefixed_login[u[User.node_id.name]],
                        name=u[User.name.name],
                        email="<classified>",  # u[User.email.name] TODO(vmarkovtsev): DEV-87
                        picture=u[User.avatar_url.name],
                        jira_user=mapped_jira.get(u[User.node_id.name]))
            for u in users
        ]
        return model_response(contributors)
