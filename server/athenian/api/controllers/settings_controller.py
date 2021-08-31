import logging
from typing import Optional

from aiohttp import web
from sqlalchemy import and_, delete, insert, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.controllers.account import get_metadata_account_ids, get_user_account_status
from athenian.api.controllers.jira import ALLOWED_USER_TYPES, get_jira_id, \
    load_jira_identity_mapping_sentinel
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.settings import ReleaseMatch, Settings
from athenian.api.models.metadata.github import User as GitHubUser
from athenian.api.models.metadata.jira import Project, User as JIRAUser
from athenian.api.models.state.models import JIRAProjectSetting, MappedJIRAIdentity
from athenian.api.models.web import ForbiddenError, InvalidRequestError, JIRAProject, \
    JIRAProjectsRequest, MappedJIRAIdentity as WebMappedJIRAIdentity, ReleaseMatchRequest, \
    ReleaseMatchSetting, SetMappedJIRAIdentitiesRequest
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
    model = {
        k: ReleaseMatchSetting.from_dataclass(m).to_dict()
        for k, m in settings.prefixed.items()
    }
    _, default_branches = await BranchMiner.extract_branches(
        settings.native, meta_ids, request.mdb, request.cache)
    for repo, name in default_branches.items():
        model[settings.prefixed_for_native(repo)]["default_branch"] = name
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
    projects = await request.mdb.fetch_all(
        select([Project.key, Project.name, Project.avatar_url])
        .where(and_(Project.acc_id == jira_id,
                    Project.is_deleted.is_(False)))
        .order_by(Project.key))
    keys = [r[Project.key.name] for r in projects]
    settings = await request.sdb.fetch_all(
        select([JIRAProjectSetting.key, JIRAProjectSetting.enabled])
        .where(and_(JIRAProjectSetting.account_id == id,
                    JIRAProjectSetting.key.in_(keys))))
    settings = {r[0]: r[1] for r in settings}
    models = [JIRAProject(name=r[Project.name.name],
                          key=r[Project.key.name],
                          avatar_url=r[Project.avatar_url.name],
                          enabled=settings.get(r[Project.key.name], True))
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
    projects = await request.mdb.fetch_all(
        select([Project.key])
        .where(and_(Project.acc_id == jira_id,
                    Project.is_deleted.is_(False))))
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
    request_model = SetMappedJIRAIdentitiesRequest.from_dict(body)
    sdb = request.sdb
    mdb = request.mdb
    cache = request.cache
    if not await get_user_account_status(request.uid, request_model.account, sdb, cache):
        raise ResponseError(ForbiddenError("You must be an admin to edit the identity mapping."))
    tasks = [
        get_jira_id(request_model.account, sdb, cache),
        get_metadata_account_ids(request_model.account, sdb, cache),
    ]
    jira_acc, meta_ids = await gather(*tasks)
    github_logins = []
    for i, c in enumerate(request_model.changes):
        try:
            github_logins.append(c.developer_id.rsplit("/", 1)[1])
        except IndexError:
            raise ResponseError(InvalidRequestError(detail="Invalid developer identifier.",
                                                    pointer=".changes[%d].developer_id" % i))
    jira_names = [c.jira_name for c in request_model.changes]
    tasks = [
        mdb.fetch_all(select([GitHubUser.node_id, GitHubUser.login])
                      .where(and_(GitHubUser.acc_id.in_(meta_ids),
                                  GitHubUser.login.in_(github_logins)))),
        mdb.fetch_all(select([JIRAUser.id, JIRAUser.display_name])
                      .where(and_(JIRAUser.acc_id == jira_acc,
                                  JIRAUser.display_name.in_(jira_names)))),
    ]
    github_id_rows, jira_id_rows = await gather(*tasks)
    github_id_map = {r[GitHubUser.login.name]: r[GitHubUser.node_id.name] for r in github_id_rows}
    jira_id_map = {r[JIRAUser.display_name.name]: r[JIRAUser.id.name] for r in jira_id_rows}
    cleared_github_ids = set()
    updated_maps = []
    for i, change in enumerate(request_model.changes):
        try:
            github_id = github_id_map[change.developer_id.rsplit("/", 1)[1]]
        except KeyError:
            raise ResponseError(InvalidRequestError(detail="Developer was not found.",
                                                    pointer=".changes[%d].developer_id" % i))
        if change.jira_name is None:
            cleared_github_ids.add(github_id)
            continue
        if not change.jira_name:
            raise ResponseError(InvalidRequestError(detail="String cannot be empty.",
                                                    pointer=".changes[%d].jira_name" % i))
        try:
            jira_id = jira_id_map[change.jira_name]
        except KeyError:
            raise ResponseError(InvalidRequestError(detail="JIRA user was not found.",
                                                    pointer=".changes[%d].jira_name" % i))
        updated_maps.append((github_id, jira_id))

    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            await sdb_conn.execute(
                delete(MappedJIRAIdentity)
                .where(and_(MappedJIRAIdentity.account_id == request_model.account,
                            MappedJIRAIdentity.github_user_id.in_(cleared_github_ids))))
            if sdb.url.dialect == "postgresql":
                sql = postgres_insert(MappedJIRAIdentity)
                sql = sql.on_conflict_do_update(
                    constraint=MappedJIRAIdentity.__table__.primary_key,
                    set_={
                        MappedJIRAIdentity.jira_user_id.name: sql.excluded.jira_user_id,
                        MappedJIRAIdentity.updated_at.name: sql.excluded.updated_at,
                        MappedJIRAIdentity.confidence.name: sql.excluded.confidence,
                    },
                )
            else:
                sql = insert(MappedJIRAIdentity).prefix_with("OR REPLACE")
            await sdb_conn.execute_many(sql, [
                MappedJIRAIdentity(
                    account_id=request_model.account,
                    github_user_id=ghid,
                    jira_user_id=jid,
                    confidence=1.0,
                ).create_defaults().explode(with_primary_keys=True) for ghid, jid in updated_maps])
    await load_jira_identity_mapping_sentinel.reset_cache(request_model.account, cache)
    return await get_jira_identities(request, request_model.account, jira_acc=jira_acc)


async def get_jira_identities(request: AthenianWebRequest,
                              id: int,
                              jira_acc: Optional[int] = None,
                              ) -> web.Response:
    """Fetch the GitHub<>JIRA user identity mapping."""
    if jira_acc is None:
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
    github_ids = [r[MappedJIRAIdentity.github_user_id.name] for r in map_rows]
    tasks = [
        request.mdb.fetch_all(
            select([GitHubUser.node_id, GitHubUser.html_url, GitHubUser.name])
            .where(and_(GitHubUser.node_id.in_(github_ids),
                        GitHubUser.acc_id.in_(meta_ids)))),
        request.mdb.fetch_all(
            select([JIRAUser.id, JIRAUser.display_name])
            .where(and_(JIRAUser.acc_id == jira_acc,
                        JIRAUser.type.in_(ALLOWED_USER_TYPES))),
        ),
    ]
    github_rows, jira_rows = await gather(*tasks)
    github_details = {r[GitHubUser.node_id.name]: r for r in github_rows}
    jira_details = {r[JIRAUser.id.name]: r for r in jira_rows}
    models = []
    log = logging.getLogger("%s.get_jira_identities" % metadata.__package__)
    mentioned_jira_user_ids = set()
    for map_row in map_rows:
        try:
            github_user = github_details[map_row[MappedJIRAIdentity.github_user_id.name]]
            jira_user = jira_details[
                (jira_user_id := map_row[MappedJIRAIdentity.jira_user_id.name])
            ]
        except KeyError:
            log.error("Identity mapping %s -> %s misses details" % (
                map_row[MappedJIRAIdentity.github_user_id.name],
                map_row[MappedJIRAIdentity.jira_user_id.name]))
            continue
        mentioned_jira_user_ids.add(jira_user_id)
        models.append(WebMappedJIRAIdentity(
            developer_id=github_user[GitHubUser.html_url.name].split("://", 1)[1],
            developer_name=github_user[GitHubUser.name.name],
            jira_name=jira_user[JIRAUser.display_name.name],
            confidence=map_row[MappedJIRAIdentity.confidence.name],
        ))
    for user_id in jira_details.keys() - mentioned_jira_user_ids:
        models.append(WebMappedJIRAIdentity(
            developer_id=None,
            developer_name=None,
            jira_name=jira_details[user_id][JIRAUser.display_name.name],
            confidence=0,
        ))
    return model_response(sorted(models, key=lambda m: (m.developer_id or "", m.jira_name)))
