from typing import Optional

from aiohttp import web
from sqlalchemy import and_, delete, insert, select

from athenian.api.async_utils import gather
from athenian.api.controllers.account import get_metadata_account_ids, get_user_account_status
from athenian.api.controllers.jira import get_jira_id
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.settings import ReleaseMatch, Settings
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.jira import Project
from athenian.api.models.state.models import JIRAProjectSetting
from athenian.api.models.web import ForbiddenError, InvalidRequestError, ReleaseMatchSetting
from athenian.api.models.web.jira_project import JIRAProject
from athenian.api.models.web.jira_projects_request import JIRAProjectsRequest
from athenian.api.models.web.release_match_request import ReleaseMatchRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


async def list_release_match_settings(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current release matching settings."""
    # Check the user separately beforehand to avoid security problems.
    await get_user_account_status(request.uid, id, request.sdb, request.cache)
    tasks = [
        get_metadata_account_ids(id, request.sdb, request.cache),
        Settings.from_request(request, id).list_release_matches(),
    ]
    meta_ids, settings = await gather(*tasks, op="sdb")
    model = {k: ReleaseMatchSetting.from_dataclass(m).to_dict() for k, m in settings.items()}
    repos = [r.split("/", 1)[1] for r in settings]
    _, default_branches = await extract_branches(repos, meta_ids, request.mdb, request.cache)
    prefix = PREFIXES["github"]
    for repo, name in default_branches.items():
        model[prefix + repo]["default_branch"] = name
    return web.json_response(model)


async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
    if request.is_default_user:
        return ResponseError(ForbiddenError("%s is the default user" % request.uid)).response
    rule = ReleaseMatchRequest.from_dict(body)
    settings = Settings.from_request(request, rule.account)
    match = ReleaseMatch[rule.match]
    repos = await settings.set_release_matches(rule.repositories, rule.branches, rule.tags, match)
    return web.json_response(sorted(repos))


async def get_jira_projects(request: AthenianWebRequest,
                            id: int,
                            jira_id: Optional[int] = None) -> web.Response:
    """List the current enabled JIRA project settings."""
    if jira_id is None:
        await get_user_account_status(request.uid, id, request.sdb, request.cache)
        jira_id = await get_jira_id(id, request.sdb, request.cache)
    projects = await request.mdb.fetch_all(select([Project.key, Project.name, Project.avatar_url])
                                           .where(Project.acc_id == jira_id)
                                           .order_by(Project.key))
    keys = [r[Project.key.key] for r in projects]
    settings = await request.sdb.fetch_all(
        select([JIRAProjectSetting.key, JIRAProjectSetting.enabled])
        .where(and_(JIRAProjectSetting.account_id == id,
                    JIRAProjectSetting.key.in_(keys))))
    settings = {r[0]: r[1] for r in settings}
    models = [JIRAProject(name=r[Project.name.key],
                          key=r[Project.key.key],
                          avatar_url=r[Project.avatar_url.key],
                          enabled=settings.get(r[Project.key.key], True))
              for r in projects]
    return model_response(models)


async def set_jira_projects(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the enabled JIRA projects."""
    model = JIRAProjectsRequest.from_dict(body)
    is_admin = await get_user_account_status(
        request.uid, model.account, request.sdb, request.cache)
    if not is_admin:
        raise ResponseError(ForbiddenError(
            detail="User %s is not an admin of account %d" % (request.uid, model.account)))
    jira_id = await get_jira_id(model.account, request.sdb, request.cache)
    projects = await request.mdb.fetch_all(select([Project.key]).where(Project.acc_id == jira_id))
    projects = {r[0] for r in projects}
    if diff := (model.projects.keys() - projects):
        raise ResponseError(InvalidRequestError(
            detail="The following JIRA projects do not exist: %s" % diff,
            pointer=".projects"))
    values = [JIRAProjectSetting(account_id=model.account,
                                 key=k,
                                 enabled=v).create_defaults().explode(with_primary_keys=True)
              for k, v in model.projects.items()]
    async with request.sdb.connection() as conn:
        async with conn.transaction():
            await conn.execute(delete(JIRAProjectSetting)
                               .where(and_(JIRAProjectSetting.account_id == model.account,
                                           JIRAProjectSetting.key.in_(projects))))
            await conn.execute_many(insert(JIRAProjectSetting), values)
    return await get_jira_projects(request, model.account, jira_id=jira_id)
