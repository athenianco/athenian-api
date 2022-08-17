from typing import Optional

from aiohttp import web

from athenian.api.align.goals.dbaccess import (
    delete_goal_template_from_db,
    dump_goal_repositories,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal_template,
    parse_goal_repositories,
    update_goal_template_in_db,
)
from athenian.api.db import integrity_errors
from athenian.api.internal.account import get_user_account_status_from_request
from athenian.api.internal.reposet import RepoIdentitiesMapper
from athenian.api.models.state.models import GoalTemplate as DBGoalTemplate
from athenian.api.models.web import (
    DatabaseConflict,
    GoalTemplate,
    GoalTemplateCreateRequest,
    GoalTemplateUpdateRequest,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


async def get_goal_template(request: AthenianWebRequest, id: int) -> web.Response:
    """Retrieve a goal template.

    :param id: Numeric identifier of the goal template.
    :type id: int
    """
    row = await get_goal_template_from_db(id, request.sdb)
    account = row[DBGoalTemplate.account_id.name]
    await get_user_account_status_from_request(request, account)

    if (db_repos := parse_goal_repositories(row[DBGoalTemplate.repositories.name])) is not None:
        mapper = await RepoIdentitiesMapper.from_request(request, account)
        repositories = mapper.identities_to_prefixed_names(db_repos)
    else:
        repositories = None
    model = GoalTemplate.from_db_row(row, repositories=repositories)
    return model_response(model)


async def list_goal_templates(request: AthenianWebRequest, id: int) -> web.Response:
    """List the goal templates for the account.

    :param id: Numeric identifier of the account.
    :type id: int
    """
    await get_user_account_status_from_request(request, id)
    rows = await get_goal_templates_from_db(id, request.sdb)

    mapper = None
    models = []
    for row in rows:
        raw_db_repos = row[DBGoalTemplate.repositories.name]
        if (db_repos := parse_goal_repositories(raw_db_repos)) is not None:
            if mapper is None:
                mapper = await RepoIdentitiesMapper.from_request(request, id)
            repositories = mapper.identities_to_prefixed_names(db_repos)
        else:
            repositories = None
        models.append(GoalTemplate.from_db_row(row, repositories=repositories))
    return model_response(models)


async def create_goal_template(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a goal template.

    :param body: GoalTemplateCreateRequest
    """
    create_request = GoalTemplateCreateRequest.from_dict(body)
    await get_user_account_status_from_request(request, create_request.account)

    repositories = await _parse_request_repositories(
        create_request.repositories, request, create_request.account,
    )

    try:
        template_id = await insert_goal_template(
            account_id=create_request.account,
            name=create_request.name,
            metric=create_request.metric,
            repositories=repositories,
            sdb=request.sdb,
        )
    except integrity_errors:
        raise ResponseError(
            DatabaseConflict(detail="Goal Template named '%s' already exists."),
        ) from None
    return model_response({"id": template_id})


async def delete_goal_template(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a goal tamplate.

    :param id: Numeric identifier of the goal template.
    """
    template = await get_goal_template_from_db(id, request.sdb)
    await get_user_account_status_from_request(request, template[DBGoalTemplate.account_id.name])
    await delete_goal_template_from_db(id, request.sdb)
    return web.json_response({})


async def update_goal_template(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Update a goal template.

    :param id: Numeric identifier of the goal template.
    :param body: GoalTemplateUpdateRequest
    """
    update_request = GoalTemplateUpdateRequest.from_dict(body)
    template = await get_goal_template_from_db(id, request.sdb)
    account_id = template[DBGoalTemplate.account_id.name]
    await get_user_account_status_from_request(request, account_id)

    repositories = await _parse_request_repositories(
        update_request.repositories, request, account_id,
    )
    values = {
        DBGoalTemplate.name.name: update_request.name,
        DBGoalTemplate.repositories.name: repositories,
    }
    await update_goal_template_in_db(id, request.sdb, **values)
    return web.json_response({})


async def _parse_request_repositories(
    repo_names: Optional[list[str]],
    request: AthenianWebRequest,
    account_id: int,
) -> Optional[list[list]]:
    if repo_names is None:
        return None
    else:
        mapper = await RepoIdentitiesMapper.from_request(request, account_id)
        return dump_goal_repositories(mapper.prefixed_names_to_identities(repo_names))
