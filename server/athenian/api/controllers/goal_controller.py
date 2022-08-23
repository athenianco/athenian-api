from typing import Optional

from aiohttp import web

from athenian.api.align.exceptions import GoalTemplateNotFoundError
from athenian.api.align.goals.dbaccess import (
    delete_goal_template_from_db,
    dump_goal_repositories,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal_template,
    parse_goal_repositories,
    update_goal_template_in_db,
)
from athenian.api.db import Row, integrity_errors
from athenian.api.internal.account import get_user_account_status_from_request
from athenian.api.internal.prefixer import Prefixer
from athenian.api.models.state.models import GoalTemplate as DBGoalTemplate
from athenian.api.models.web import (
    CreatedIdentifier,
    DatabaseConflict,
    GoalTemplate,
    GoalTemplateCreateRequest,
    GoalTemplateUpdateRequest,
    InvalidRequestError,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


def _goal_template_from_row(row: Row, **kwargs) -> GoalTemplate:
    return GoalTemplate(
        id=row[DBGoalTemplate.id.name],
        name=row[DBGoalTemplate.name.name],
        metric=row[DBGoalTemplate.metric.name],
        **kwargs,
    )


async def get_goal_template(request: AthenianWebRequest, id: int) -> web.Response:
    """Retrieve a goal template.

    :param id: Numeric identifier of the goal template.
    :type id: int
    """
    row = await get_goal_template_from_db(id, request.sdb)
    account = row[DBGoalTemplate.account_id.name]
    try:
        await get_user_account_status_from_request(request, account)
    except ResponseError:
        # do not leak the account that owns this template
        raise GoalTemplateNotFoundError(id)
    if (db_repos := parse_goal_repositories(row[DBGoalTemplate.repositories.name])) is not None:
        prefixer = await Prefixer.from_request(request, account)
        repositories = prefixer.repo_identities_to_prefixed_names(db_repos)
    else:
        repositories = None
    model = _goal_template_from_row(row, repositories=repositories)
    return model_response(model)


async def list_goal_templates(request: AthenianWebRequest, id: int) -> web.Response:
    """List the goal templates for the account.

    :param id: Numeric identifier of the account.
    :type id: int
    """
    await get_user_account_status_from_request(request, id)
    rows = await get_goal_templates_from_db(id, request.sdb)

    prefixer = await Prefixer.from_request(request, id)
    models = []
    for row in rows:
        raw_db_repos = row[DBGoalTemplate.repositories.name]
        if (db_repos := parse_goal_repositories(raw_db_repos)) is not None:
            repositories = prefixer.repo_identities_to_prefixed_names(db_repos)
        else:
            repositories = None
        models.append(_goal_template_from_row(row, repositories=repositories))
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
    values = {
        DBGoalTemplate.account_id.name: create_request.account,
        DBGoalTemplate.name.name: create_request.name,
        DBGoalTemplate.metric.name: create_request.metric,
        DBGoalTemplate.repositories.name: repositories,
    }
    try:
        template_id = await insert_goal_template(request.sdb, **values)
    except integrity_errors:
        raise ResponseError(
            DatabaseConflict(
                detail=f"Goal template named '{create_request.name}' already exists.",
            ),
        ) from None
    return model_response(CreatedIdentifier(template_id))


async def delete_goal_template(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a goal tamplate.

    :param id: Numeric identifier of the goal template.
    """
    template = await get_goal_template_from_db(id, request.sdb)
    try:
        await get_user_account_status_from_request(
            request, template[DBGoalTemplate.account_id.name],
        )
    except ResponseError:
        raise GoalTemplateNotFoundError(id) from None
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
    try:
        await get_user_account_status_from_request(
            request, template[DBGoalTemplate.account_id.name],
        )
    except ResponseError:
        raise GoalTemplateNotFoundError(id) from None
    repositories = await _parse_request_repositories(
        update_request.repositories, request, account_id,
    )
    values = {
        DBGoalTemplate.name.name: update_request.name,
        DBGoalTemplate.metric.name: update_request.metric,
        DBGoalTemplate.repositories.name: repositories,
    }
    await update_goal_template_in_db(id, request.sdb, **values)
    return web.json_response({})


async def _parse_request_repositories(
    repo_names: Optional[list[str]],
    request: AthenianWebRequest,
    account_id: int,
) -> Optional[list[tuple[int, str]]]:
    if repo_names is None:
        return None
    prefixer = await Prefixer.from_request(request, account_id)
    try:
        return dump_goal_repositories(prefixer.prefixed_repo_names_to_identities(repo_names))
    except ValueError as e:
        raise ResponseError(InvalidRequestError(".repositories", str(e)))
