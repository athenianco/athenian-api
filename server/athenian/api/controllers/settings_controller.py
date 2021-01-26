import logging
from typing import Optional

from aiohttp import web
from sqlalchemy import and_, delete, insert, select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.controllers.account import get_metadata_account_ids, get_user_account_status
from athenian.api.controllers.jira import get_jira_id
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.settings import ReleaseMatch, Settings
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import User as GitHubUser
from athenian.api.models.metadata.jira import Project, User as JIRAUser
from athenian.api.models.state.models import JIRAProjectSetting, MappedJIRAIdentity
from athenian.api.models.web import ForbiddenError, InvalidRequestError, JIRAProject, \
    JIRAProjectsRequest, MappedJIRAIdentity as WebMappedJIRAIdentity, ReleaseMatchRequest, \
    ReleaseMatchSetting
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


@disable_default_user
async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
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


@disable_default_user
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


async def set_jira_identities(request: AthenianWebRequest, body: dict) -> web.Response:
    """Add or override GitHub<>JIRA user identity mapping."""
    raise NotImplementedError


async def get_jira_identities(request: AthenianWebRequest, id: int) -> web.Response:
    """Fetch the GitHub<>JIRA user identity mapping."""
    await get_user_account_status(request.uid, id, request.sdb, request.cache)
    jira_acc = await get_jira_id(id, request.sdb, request.cache)
    tasks = [
        request.sdb.fetch_all(
            select([MappedJIRAIdentity.github_user_id, MappedJIRAIdentity.jira_user_id,
                    MappedJIRAIdentity.confidence])
            .where(MappedJIRAIdentity.account_id == id)),
        get_metadata_account_ids(id, request.sdb, request.cache),
    ]
    map_rows, meta_ids = await gather(*tasks)
    github_ids = [r[MappedJIRAIdentity.github_user_id.key] for r in map_rows]
    jira_ids = [r[MappedJIRAIdentity.jira_user_id.key] for r in map_rows]
    tasks = [
        request.mdb.fetch_all(
            select([GitHubUser.node_id, GitHubUser.login, GitHubUser.name])
            .where(and_(GitHubUser.node_id.in_(github_ids),
                        GitHubUser.acc_id.in_(meta_ids)))),
        request.mdb.fetch_all(
            select([JIRAUser.id, JIRAUser.display_name])
            .where(and_(JIRAUser.id.in_(jira_ids),
                        JIRAUser.acc_id == jira_acc)),
        ),
    ]
    github_rows, jira_rows = await gather(*tasks)
    github_details = {r[GitHubUser.node_id.key]: r for r in github_rows}
    jira_details = {r[JIRAUser.id.key]: r for r in jira_rows}
    models = []
    prefix = PREFIXES["github"]
    log = logging.getLogger("%s.get_jira_identities" % metadata.__package__)
    for map_row in map_rows:
        try:
            github_user = github_details[map_row[MappedJIRAIdentity.github_user_id.key]]
            jira_user = jira_details[map_row[MappedJIRAIdentity.jira_user_id.key]]
        except KeyError:
            log.error("Identity mapping %s -> %s misses details" % (
                map_row[MappedJIRAIdentity.github_user_id.key],
                map_row[MappedJIRAIdentity.jira_user_id.key]))
            continue
        models.append(WebMappedJIRAIdentity(
            developer_id=prefix + github_user[GitHubUser.login.key],
            developer_name=github_user[GitHubUser.name.key],
            jira_name=jira_user[JIRAUser.display_name.key],
            confidence=map_row[MappedJIRAIdentity.confidence.key],
        ))
    return model_response(sorted(models, key=lambda m: m.developer_id))
