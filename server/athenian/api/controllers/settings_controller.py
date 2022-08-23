from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
import logging
import re
from typing import Optional, Sequence, Tuple

from aiohttp import web
from morcilla import Connection
from sqlalchemy import and_, delete, distinct, func, insert, select, union, update

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.db import Database, DatabaseLike, dialect_specific_insert
from athenian.api.internal.account import (
    get_metadata_account_ids,
    get_user_account_status_from_request,
    only_admin,
)
from athenian.api.internal.jira import (
    ALLOWED_USER_TYPES,
    get_jira_id,
    load_jira_identity_mapping_sentinel,
)
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import ReleaseMatch, Settings
from athenian.api.models.metadata.github import User as GitHubUser
from athenian.api.models.metadata.jira import Issue, Project, User as JIRAUser
from athenian.api.models.precomputed.models import (
    GitHubCommitDeployment,
    GitHubDeploymentFacts,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubOpenPullRequestFacts,
    GitHubPullRequestDeployment,
    GitHubRelease,
    GitHubReleaseDeployment,
    GitHubReleaseFacts,
    GitHubReleaseMatchTimespan,
)
from athenian.api.models.state.models import (
    JIRAProjectSetting,
    LogicalRepository,
    MappedJIRAIdentity,
    ReleaseSetting,
    RepositorySet,
    WorkType,
)
from athenian.api.models.web import (
    BadRequestError,
    ForbiddenError,
    InvalidRequestError,
    JIRAProject,
    JIRAProjectsRequest,
    LogicalDeploymentRules,
    LogicalPRRules,
    LogicalRepository as WebLogicalRepository,
    LogicalRepositoryGetRequest,
    LogicalRepositoryRequest,
    MappedJIRAIdentity as WebMappedJIRAIdentity,
    NotFoundError,
    ReleaseMatchRequest,
    ReleaseMatchSetting,
    SetMappedJIRAIdentitiesRequest,
    WorkType as WebWorkType,
    WorkTypeGetRequest,
    WorkTypePutRequest,
    WorkTypeRule,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


async def list_release_match_settings(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current release matching settings."""
    # Check the user separately beforehand to avoid security problems.
    await get_user_account_status_from_request(request, id)

    async def load_prefixer():
        meta_ids = await get_metadata_account_ids(id, request.sdb, request.cache)
        prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
        return meta_ids, prefixer

    (meta_ids, prefixer), settings = await gather(
        load_prefixer(), Settings.from_request(request, id).list_release_matches(), op="sdb",
    )
    model = {
        k: ReleaseMatchSetting.from_dataclass(m).to_dict() for k, m in settings.prefixed.items()
    }
    _, default_branches = await BranchMiner.extract_branches(
        settings.native, prefixer, meta_ids, request.mdb, request.cache,
    )
    unresolved = []
    for repo, name in default_branches.items():
        try:
            model[settings.prefixed_for_native(repo)]["default_branch"] = name
        except KeyError:
            unresolved.append(repo)
    if unresolved:
        log = logging.getLogger(f"{metadata.__package__}.list_release_match_settings")
        log.error("unresolved repositories in release settings: %s", unresolved)
    return web.json_response(model)


@disable_default_user
async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
    try:
        rule = ReleaseMatchRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(BadRequestError(str(e))) from e
    settings = Settings.from_request(request, rule.account)
    match = ReleaseMatch[rule.match]
    repos = await settings.set_release_matches(
        rule.repositories, rule.branches, rule.tags, rule.events or ".*", match,
    )
    return web.json_response(sorted(repos))


@weight(0.5)
async def get_jira_projects(
    request: AthenianWebRequest,
    id: int,
    jira_id: Optional[int] = None,
) -> web.Response:
    """List the current enabled JIRA project settings."""
    mdb, sdb = request.mdb, request.sdb
    if jira_id is None:
        await get_user_account_status_from_request(request, id)
        jira_id = await get_jira_id(id, sdb, request.cache)
    projects, stats = await gather(
        mdb.fetch_all(
            select([Project.key, Project.id, Project.name, Project.avatar_url])
            .where(and_(Project.acc_id == jira_id, Project.is_deleted.is_(False)))
            .order_by(Project.key),
        ),
        mdb.fetch_all(
            select(
                [
                    Issue.project_id,
                    func.count(Issue.id).label("issues_count"),
                    func.max(Issue.updated).label("last_update"),
                ],
            )
            .where(Issue.acc_id == jira_id)
            .group_by(Issue.project_id),
        ),
        op="get_jira_projects/mdb",
    )
    stats = {r[Issue.project_id.name]: (r["issues_count"], r["last_update"]) for r in stats}
    keys = [r[Project.key.name] for r in projects]
    settings = dict(
        await sdb.fetch_all(
            select([JIRAProjectSetting.key, JIRAProjectSetting.enabled]).where(
                and_(JIRAProjectSetting.account_id == id, JIRAProjectSetting.key.in_(keys)),
            ),
        ),
    )
    models = [
        JIRAProject(
            name=r[Project.name.name],
            key=r[Project.key.name],
            id=(pid := r[Project.id.name]),
            avatar_url=r[Project.avatar_url.name],
            enabled=settings.get(r[Project.key.name], True),
            issues_count=(rs := stats.get(pid, (0, None)))[0],
            last_update=rs[1],
        )
        for r in projects
    ]
    if mdb.url.dialect == "sqlite":
        for m in models:
            if m.last_update is not None:
                m.last_update = m.last_update.replace(tzinfo=timezone.utc)
    return model_response(models)


@disable_default_user
@only_admin
async def set_jira_projects(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the enabled JIRA projects."""
    model = JIRAProjectsRequest.from_dict(body)
    jira_id = await get_jira_id(model.account, request.sdb, request.cache)
    projects = await request.mdb.fetch_all(
        select([Project.key]).where(
            and_(Project.acc_id == jira_id, Project.is_deleted.is_(False)),
        ),
    )
    projects = {r[0] for r in projects}
    if diff := (model.projects.keys() - projects):
        raise ResponseError(
            InvalidRequestError(
                detail="The following JIRA projects do not exist: %s" % diff, pointer=".projects",
            ),
        )
    values = [
        JIRAProjectSetting(account_id=model.account, key=k, enabled=v)
        .create_defaults()
        .explode(with_primary_keys=True)
        for k, v in model.projects.items()
    ]
    async with request.sdb.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                delete(JIRAProjectSetting).where(
                    and_(
                        JIRAProjectSetting.account_id == model.account,
                        JIRAProjectSetting.key.in_(projects),
                    ),
                ),
            )
            await conn.execute_many(insert(JIRAProjectSetting), values)
    return await get_jira_projects(request, model.account, jira_id=jira_id)


@only_admin
async def set_jira_identities(request: AthenianWebRequest, body: dict) -> web.Response:
    """Add or override GitHub<>JIRA user identity mapping."""
    request_model = SetMappedJIRAIdentitiesRequest.from_dict(body)
    sdb = request.sdb
    mdb = request.mdb
    cache = request.cache
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
            raise ResponseError(
                InvalidRequestError(
                    detail="Invalid developer identifier.",
                    pointer=".changes[%d].developer_id" % i,
                ),
            )
    jira_names = [c.jira_name for c in request_model.changes]
    tasks = [
        mdb.fetch_all(
            select([GitHubUser.node_id, GitHubUser.login]).where(
                and_(GitHubUser.acc_id.in_(meta_ids), GitHubUser.login.in_(github_logins)),
            ),
        ),
        mdb.fetch_all(
            select([JIRAUser.id, JIRAUser.display_name]).where(
                and_(
                    JIRAUser.acc_id == jira_acc,
                    JIRAUser.type.in_(ALLOWED_USER_TYPES),
                    JIRAUser.display_name.in_(jira_names),
                ),
            ),
        ),
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
            raise ResponseError(
                InvalidRequestError(
                    detail="Developer was not found.", pointer=".changes[%d].developer_id" % i,
                ),
            )
        if change.jira_name is None:
            cleared_github_ids.add(github_id)
            continue
        if not change.jira_name:
            raise ResponseError(
                InvalidRequestError(
                    detail="String cannot be empty.", pointer=".changes[%d].jira_name" % i,
                ),
            )
        try:
            jira_id = jira_id_map[change.jira_name]
        except KeyError:
            raise ResponseError(
                InvalidRequestError(
                    detail="JIRA user was not found.", pointer=".changes[%d].jira_name" % i,
                ),
            )
        updated_maps.append((github_id, jira_id))

    try:
        async with sdb.connection() as sdb_conn:
            async with sdb_conn.transaction():
                await sdb_conn.execute(
                    delete(MappedJIRAIdentity).where(
                        and_(
                            MappedJIRAIdentity.account_id == request_model.account,
                            MappedJIRAIdentity.github_user_id.in_(cleared_github_ids),
                        ),
                    ),
                )
                sql = (await dialect_specific_insert(sdb))(MappedJIRAIdentity)
                sql = sql.on_conflict_do_update(
                    index_elements=MappedJIRAIdentity.__table__.primary_key.columns,
                    set_={
                        MappedJIRAIdentity.jira_user_id.name: sql.excluded.jira_user_id,
                        MappedJIRAIdentity.updated_at.name: sql.excluded.updated_at,
                        MappedJIRAIdentity.confidence.name: sql.excluded.confidence,
                    },
                )
                await sdb_conn.execute_many(
                    sql,
                    [
                        MappedJIRAIdentity(
                            account_id=request_model.account,
                            github_user_id=ghid,
                            jira_user_id=jid,
                            confidence=1.0,
                        )
                        .create_defaults()
                        .explode(with_primary_keys=True)
                        for ghid, jid in updated_maps
                    ],
                )
    finally:
        await load_jira_identity_mapping_sentinel.reset_cache(request_model.account, cache)
    return await get_jira_identities(request, request_model.account, jira_acc=jira_acc)


async def get_jira_identities(
    request: AthenianWebRequest,
    id: int,
    jira_acc: Optional[int] = None,
) -> web.Response:
    """Fetch the GitHub<>JIRA user identity mapping."""
    if jira_acc is None:
        await get_user_account_status_from_request(request, id)
        jira_acc = await get_jira_id(id, request.sdb, request.cache)
    tasks = [
        request.sdb.fetch_all(
            select(
                [
                    MappedJIRAIdentity.github_user_id,
                    MappedJIRAIdentity.jira_user_id,
                    MappedJIRAIdentity.confidence,
                ],
            ).where(MappedJIRAIdentity.account_id == id),
        ),
        get_metadata_account_ids(id, request.sdb, request.cache),
    ]
    map_rows, meta_ids = await gather(*tasks)
    github_ids = [r[MappedJIRAIdentity.github_user_id.name] for r in map_rows]
    tasks = [
        request.mdb.fetch_all(
            select([GitHubUser.node_id, GitHubUser.html_url, GitHubUser.name]).where(
                and_(GitHubUser.node_id.in_(github_ids), GitHubUser.acc_id.in_(meta_ids)),
            ),
        ),
        request.mdb.fetch_all(
            select([JIRAUser.id, JIRAUser.display_name]).where(
                and_(JIRAUser.acc_id == jira_acc, JIRAUser.type.in_(ALLOWED_USER_TYPES)),
            ),
        ),
    ]
    github_rows, jira_rows = await gather(*tasks)
    github_details = {r[GitHubUser.node_id.name]: r for r in github_rows}
    jira_details = {r[JIRAUser.id.name]: r for r in jira_rows}
    models = []
    log = logging.getLogger(f"{metadata.__package__}.get_jira_identities")
    mentioned_jira_user_ids = set()
    for map_row in map_rows:
        try:
            github_user = github_details[map_row[MappedJIRAIdentity.github_user_id.name]]
        except KeyError:
            log.error(
                "Identity mapping %s -> %s misses GitHub details",
                map_row[MappedJIRAIdentity.github_user_id.name],
                map_row[MappedJIRAIdentity.jira_user_id.name],
            )
            continue
        try:
            jira_user = jira_details[
                (jira_user_id := map_row[MappedJIRAIdentity.jira_user_id.name])
            ]
        except KeyError:
            log.warning(
                "Identity mapping %s -> %s misses JIRA details",
                map_row[MappedJIRAIdentity.github_user_id.name],
                map_row[MappedJIRAIdentity.jira_user_id.name],
            )
            continue
        mentioned_jira_user_ids.add(jira_user_id)
        models.append(
            WebMappedJIRAIdentity(
                developer_id=github_user[GitHubUser.html_url.name].split("://", 1)[1],
                developer_name=github_user[GitHubUser.name.name],
                jira_name=jira_user[JIRAUser.display_name.name],
                confidence=map_row[MappedJIRAIdentity.confidence.name],
            ),
        )
    for user_id in jira_details.keys() - mentioned_jira_user_ids:
        models.append(
            WebMappedJIRAIdentity(
                developer_id=None,
                developer_name=None,
                jira_name=jira_details[user_id][JIRAUser.display_name.name],
                confidence=0,
            ),
        )
    return model_response(sorted(models, key=lambda m: (m.developer_id or "", m.jira_name)))


async def get_work_type(request: AthenianWebRequest, body: dict) -> web.Response:
    """Fetch the definition of the work type given the name."""
    model = WorkTypeGetRequest.from_dict(body)
    row = await request.sdb.fetch_one(
        select([WorkType]).where(
            and_(
                WorkType.account_id == model.account,
                WorkType.name == model.name,
            ),
        ),
    )
    if row is None:
        raise ResponseError(NotFoundError(f'Work type "{model.name}" does not exist.'))
    model = WebWorkType(
        name=row[WorkType.name.name],
        color=row[WorkType.color.name],
        rules=[WorkTypeRule(name, args) for name, args in row[WorkType.rules.name]],
    )
    return model_response(model)


async def set_work_type(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create or update a work type - a rule set to group PRs, releases, etc. together."""
    model = WorkTypePutRequest.from_dict(body)
    sdb = request.sdb
    sql = (await dialect_specific_insert(sdb))(WorkType)
    sql = sql.on_conflict_do_update(
        index_elements=[WorkType.account_id, WorkType.name],
        set_={
            col.name: getattr(sql.excluded, col.name)
            for col in (WorkType.color, WorkType.rules, WorkType.updated_at)
        },
    )
    await sdb.execute(
        sql.values(
            WorkType(
                account_id=model.account,
                name=model.work_type.name,
                color=model.work_type.color,
                rules=[(r.name, r.body) for r in model.work_type.rules],
            )
            .create_defaults()
            .explode(),
        ),
    )
    return model_response(model.work_type)


async def delete_work_type(request: AthenianWebRequest, body: dict) -> web.Response:
    """Remove the work type given the name."""
    model = WorkTypeGetRequest.from_dict(body)
    row = await request.sdb.fetch_one(
        select([WorkType]).where(
            and_(
                WorkType.account_id == model.account,
                WorkType.name == model.name,
            ),
        ),
    )
    if row is None:
        raise ResponseError(NotFoundError(f'Work type "{model.name}" does not exist.'))
    await request.sdb.execute(
        delete(WorkType).where(
            and_(
                WorkType.account_id == model.account,
                WorkType.name == model.name,
            ),
        ),
    )
    return web.Response()


async def list_work_types(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current work types - rule sets to group PRs, releases, etc. together."""
    account = id
    await get_user_account_status_from_request(request, account)
    rows = await request.sdb.fetch_all(select([WorkType]).where(WorkType.account_id == account))
    models = [
        WebWorkType(
            name=row[WorkType.name.name],
            color=row[WorkType.color.name],
            rules=[WorkTypeRule(name, args) for name, args in row[WorkType.rules.name]],
        )
        for row in rows
    ]
    return model_response(models)


async def list_logical_repositories(request: AthenianWebRequest, id: int) -> web.Response:
    """List the currently configured logical repositories."""
    await get_user_account_status_from_request(request, id)
    settings = Settings.from_request(request, id)

    async def load_prefixer():
        meta_ids = await get_metadata_account_ids(id, request.sdb, request.cache)
        return await Prefixer.load(meta_ids, request.mdb, request.cache)

    prefixer, release_settings, rows = await gather(
        load_prefixer(),
        settings.list_release_matches(),
        request.sdb.fetch_all(
            select([LogicalRepository]).where(LogicalRepository.account_id == id),
        ),
    )
    models = []
    for row in rows:
        repo = prefixer.repo_node_to_name[row[LogicalRepository.repository_id.name]]
        prefixed_repo = prefixer.repo_name_to_prefixed_name[repo]
        name = row[LogicalRepository.name.name]
        full_name = f"{repo}/{name}"
        prs = row[LogicalRepository.prs.name]
        deployments = row[LogicalRepository.deployments.name]
        models.append(
            WebLogicalRepository(
                name=name,
                parent=prefixed_repo,
                prs=LogicalPRRules(
                    title=prs.get("title"),
                    labels_include=prs.get("labels"),
                ),
                releases=ReleaseMatchSetting.from_dataclass(release_settings.native[full_name]),
                deployments=LogicalDeploymentRules(
                    title=deployments.get("title"),
                    labels_include=deployments.get("labels"),
                ),
            ),
        )
    return model_response(models)


@only_admin
async def set_logical_repository(request: AthenianWebRequest, body: dict) -> web.Response:
    """Insert or update a logical repository."""
    web_model = LogicalRepositoryRequest.from_dict(body)
    meta_ids = await get_metadata_account_ids(web_model.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    try:
        repo = web_model.parent.split("/", 1)[1]
    except IndexError as e:
        raise ResponseError(
            InvalidRequestError(
                ".parent",
                "Repository name must be prefixed (e.g. `github.com/athenianco/athenian-api`).",
            ),
        ) from e
    try:
        repo_id = prefixer.repo_name_to_node[repo]
    except KeyError:
        raise ResponseError(
            ForbiddenError(f"Access denied to `{web_model.parent}` or it does not exist."),
        )
    db_model = LogicalRepository(
        account_id=web_model.account,
        name=web_model.name,
        repository_id=repo_id,
        prs={"title": web_model.prs.title, "labels": web_model.prs.labels_include},
    ).create_defaults()
    if (deployments := web_model.deployments) is not None:
        db_model.deployments = {"title": deployments.title, "labels": deployments.labels_include}
    settings = Settings.from_request(request, web_model.account)
    full_name = f"{repo}/{web_model.name}"
    prefixed_name = f"{web_model.parent}/{web_model.name}"

    del body["account"]
    response = model_response(WebLogicalRepository.from_dict(body))

    async with request.sdb.connection() as sdb_conn:
        existing, equal = await _find_matching_logical_repository(
            db_model, web_model.parent, web_model.releases, sdb_conn,
        )
        if existing:
            if equal:
                # equal logical repository exists, nothing to do
                return response
            else:
                # existing logical repository is different, delete it
                async with sdb_conn.transaction():
                    await _delete_logical_repository(
                        web_model.name,
                        full_name,
                        prefixed_name,
                        repo_id,
                        web_model.account,
                        sdb_conn,
                        request.pdb,
                    )

        else:
            # new logical repo invalidates logical deployments
            await _clean_logical_deployments(
                [repo], web_model.account, request.pdb, op="_clean_logical_deployments/sql",
            )
        async with sdb_conn.transaction():
            settings._sdb = sdb_conn
            # create the logical repository setting
            await sdb_conn.execute(insert(LogicalRepository).values(db_model.explode()))
            # create the release settings
            await settings.set_release_matches(
                [prefixed_name],
                web_model.releases.branches,
                web_model.releases.tags,
                web_model.releases.events,
                ReleaseMatch[web_model.releases.match],
                dereference=False,
            )
            # append to repository sets
            rows = await sdb_conn.fetch_all(
                select([RepositorySet]).where(RepositorySet.owner_id == web_model.account),
            )
            for row in rows:
                if re.fullmatch(row[RepositorySet.tracking_re.name], prefixed_name):
                    items = row[RepositorySet.items.name]
                    items.insert(
                        bisect_right([r[0] for r in items], prefixed_name),
                        (prefixed_name, repo_id),
                    )
                    await sdb_conn.execute(
                        update(RepositorySet)
                        .where(RepositorySet.id == row[RepositorySet.id.name])
                        .values(
                            {
                                RepositorySet.updates_count: RepositorySet.updates_count + 1,
                                RepositorySet.updated_at: datetime.now(timezone.utc),
                                RepositorySet.items: items,
                            },
                        ),
                    )
    return response


async def _find_matching_logical_repository(
    logical_repo: LogicalRepository,
    parent_name: str,
    release_match_setting: ReleaseMatchSetting,
    sdb_conn: Connection,
) -> Tuple[Optional[LogicalRepository], bool]:
    """Find a logical repository matching `logical_repo`.

    Properties matched are `account_id`, `name` and parent `repository_id`.

    Return the matched logical repo row, if any, and a boolean telling if the
    existing logical repo is identical to the compared logical repo also considering
    extra properties and `release_match_setting`.
    """
    existing = await sdb_conn.fetch_one(
        select([LogicalRepository]).where(
            and_(
                LogicalRepository.account_id == logical_repo.account_id,
                LogicalRepository.name == logical_repo.name,
                LogicalRepository.repository_id == logical_repo.repository_id,
            ),
        ),
    )
    if existing is None:
        return None, False
    for col in (LogicalRepository.prs, LogicalRepository.deployments):
        if existing[col.name] != getattr(logical_repo, col.name):
            return existing, False

    matching_release_setting = await sdb_conn.fetch_one(
        select(ReleaseSetting).where(
            ReleaseSetting.repository == f"{parent_name}/{logical_repo.name}",
            ReleaseSetting.account_id == logical_repo.account_id,
            ReleaseSetting.branches == release_match_setting.branches,
            ReleaseSetting.tags == release_match_setting.tags,
            ReleaseSetting.events == release_match_setting.events,
            ReleaseSetting.match == ReleaseMatch[release_match_setting.match],
        ),
    )
    return existing, matching_release_setting is not None


async def _delete_logical_repository(
    name: str,
    full_name: str,
    prefixed_name: str,
    repo_id: int,
    account: int,
    sdb: DatabaseLike,
    pdb: Database,
) -> None:
    async def clean_repository_sets():
        rows = await sdb.fetch_all(
            select([RepositorySet]).where(RepositorySet.owner_id == account),
        )
        for row in rows:
            items = row[RepositorySet.items.name]
            index = bisect_left([r[0] for r in items], prefixed_name)
            if index < len(items) and items[index][0] == prefixed_name:
                items.pop(index)
                await sdb.execute(
                    update(RepositorySet)
                    .where(RepositorySet.id == row[RepositorySet.id.name])
                    .values(
                        {
                            RepositorySet.updates_count: RepositorySet.updates_count + 1,
                            RepositorySet.updated_at: datetime.now(timezone.utc),
                            RepositorySet.items: items,
                        },
                    ),
                )

    tasks = [
        sdb.execute(
            delete(LogicalRepository).where(
                and_(
                    LogicalRepository.account_id == account,
                    LogicalRepository.name == name,
                    LogicalRepository.repository_id == repo_id,
                ),
            ),
        ),
        sdb.execute(
            delete(ReleaseSetting).where(
                and_(
                    ReleaseSetting.account_id == account,
                    ReleaseSetting.repository == prefixed_name,
                ),
            ),
        ),
        clean_repository_sets(),
    ]
    physical_repo = drop_logical_repo(full_name)
    for model in (
        GitHubDonePullRequestFacts,
        GitHubMergedPullRequestFacts,
        GitHubOpenPullRequestFacts,
        GitHubRelease,
        GitHubReleaseFacts,
        GitHubReleaseMatchTimespan,
    ):
        tasks.append(
            pdb.execute(
                delete(model).where(
                    and_(
                        model.acc_id == account,
                        model.repository_full_name.in_([full_name, physical_repo]),
                    ),
                ),
            ),
        )
    clean_deployments_coro = _clean_logical_deployments([full_name, physical_repo], account, pdb)
    await gather(*tasks, clean_deployments_coro, op="_delete_logical_repository/sql")


async def _clean_logical_deployments(
    repos: Sequence[str],
    account: int,
    pdb: DatabaseLike,
    op: str = None,
) -> None:
    """Delete all deployments info about `repos` from pdb."""
    deployment_models = (
        GitHubCommitDeployment,
        GitHubPullRequestDeployment,
        GitHubReleaseDeployment,
    )
    affected_deployments = await pdb.fetch_all(
        union(
            *(
                select([distinct(model.deployment_name)]).where(
                    and_(
                        model.acc_id == account,
                        model.repository_full_name.in_(repos),
                    ),
                )
                for model in deployment_models
            ),
        ),
    )
    tasks = []
    if affected_deployments:
        affected_deployment_names = [r[0] for r in affected_deployments]
        log = logging.getLogger(f"{metadata.__package__}._clean_logical_deployments")
        log.info("Cleaning deployments %s for repos %s", affected_deployment_names, repos)
        for model in deployment_models:
            delete_stmt = delete(model).where(
                and_(model.acc_id == account, model.repository_full_name.in_(repos)),
            )
            tasks.append(pdb.execute(delete_stmt))
        facts_delete_stmt = delete(GitHubDeploymentFacts).where(
            and_(
                GitHubDeploymentFacts.acc_id == account,
                GitHubDeploymentFacts.deployment_name.in_(affected_deployment_names),
            ),
        )
        tasks.append(pdb.execute(facts_delete_stmt))
    await gather(*tasks, op=op)


@only_admin
async def delete_logical_repository(request: AthenianWebRequest, body: dict) -> web.Response:
    """Delete a logical repository."""
    model = LogicalRepositoryGetRequest.from_dict(body)
    try:
        prefix, org, name, logical = model.name.split("/", 3)
    except ValueError:
        raise ResponseError(InvalidRequestError(".name", "Invalid logical repository name."))
    root_name = f"{org}/{name}"
    meta_ids = await get_metadata_account_ids(model.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    try:
        repo_id = prefixer.repo_name_to_node[root_name]
    except KeyError:
        raise ResponseError(
            ForbiddenError(f"Access denied to `{prefix}/{root_name}` or it does not exist."),
        )
    repo = await request.sdb.fetch_one(
        select([LogicalRepository]).where(
            and_(
                LogicalRepository.account_id == model.account,
                LogicalRepository.name == logical,
                LogicalRepository.repository_id == repo_id,
            ),
        ),
    )
    if repo is None:
        raise ResponseError(NotFoundError(f"Logical repository {model.name} does not exist."))
    await _delete_logical_repository(
        logical,
        f"{org}/{name}/{logical}",
        model.name,
        repo_id,
        model.account,
        request.sdb,
        request.pdb,
    )
    return web.Response()
