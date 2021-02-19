from aiohttp import web
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.settings import ReleaseMatch, Settings
from athenian.api.models.metadata.github import PushCommit
from athenian.api.models.web import ForbiddenError, InvalidRequestError, NotifyReleaseRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError


@weight(0)
async def notify_release(request: AthenianWebRequest, body: dict) -> web.Response:
    """Notify about new releases. The release settings must be set to "notification"."""
    # account is automatically checked at this point
    notifications = NotifyReleaseRequest.from_dict(body)
    account = notifications.account
    repos = {n.repository for n in notifications.notifications}
    tasks = [
        Settings.from_request(request, account).list_release_matches(repos),
        get_metadata_account_ids(account, request.sdb, request.cache),
    ]
    release_settings, meta_ids = await gather(*tasks)
    # check that release_settings do not contradict with notifications
    if repos_diff := repos - release_settings.keys():
        raise ResponseError(ForbiddenError(
            "The following repositories do not belong to the account %d: %s" %
            (account, repos_diff)))
    if wrong_settings := [repo for repo, rs in release_settings.items()
                          if rs.match != ReleaseMatch.notification]:
        raise ResponseError(InvalidRequestError(
            pointer=".notifications",
            detail="The following repositories do not allow release notifications: %s" %
                   wrong_settings))
    mdb = request.mdb
    # the commit may not exist yet in the metadata, but let's check
    await mdb.fetch_one(select([PushCommit.sha, PushCommit.node_id]).where(and_(
        PushCommit.acc_id.in_(meta_ids),
        PushCommit.sha.in_([]),
    )))
    raise NotImplementedError
