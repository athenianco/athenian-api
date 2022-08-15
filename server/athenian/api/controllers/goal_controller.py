from aiohttp import web

from athenian.api.align.exceptions import GoalTemplateNotFoundError
from athenian.api.align.goals.dbaccess import (
    delete_goal_template_from_db,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal_template,
    update_goal_template_in_db,
)
from athenian.api.db import integrity_errors
from athenian.api.internal.account import get_user_account_status_from_request
from athenian.api.models.state.models import GoalTemplate as DBGoalTemplate
from athenian.api.models.web import (
    CreatedIdentifier,
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
    try:
        await get_user_account_status_from_request(request, row[DBGoalTemplate.account_id.name])
    except ResponseError:
        # do not leak the account that owns this template
        raise GoalTemplateNotFoundError(id)
    model = GoalTemplate(
        id=id, name=row[DBGoalTemplate.name.name], metric=row[DBGoalTemplate.metric.name],
    )

    return model_response(model)


async def list_goal_templates(request: AthenianWebRequest, id: int) -> web.Response:
    """List the goal templates for the account.

    :param id: Numeric identifier of the account.
    :type id: int
    """
    await get_user_account_status_from_request(request, id)
    rows = await get_goal_templates_from_db(id, request.sdb)
    models = [
        GoalTemplate(
            id=row[DBGoalTemplate.id.name],
            name=row[DBGoalTemplate.name.name],
            metric=row[DBGoalTemplate.metric.name],
        )
        for row in rows
    ]
    return model_response(models)


async def create_goal_template(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a goal template.

    :param body: GoalTemplateCreateRequest
    """
    create_request = GoalTemplateCreateRequest.from_dict(body)
    await get_user_account_status_from_request(request, create_request.account)
    try:
        template_id = await insert_goal_template(
            create_request.account, create_request.name, create_request.metric, request.sdb,
        )
    except integrity_errors:
        raise ResponseError(
            DatabaseConflict(
                detail=f"Goal Template named '{create_request.name}' already exists.",
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
    return web.json_response()


async def update_goal_template(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Update a goal template.

    :param id: Numeric identifier of the goal template.
    :param body: GoalTemplateUpdateRequest
    """
    update_request = GoalTemplateUpdateRequest.from_dict(body)
    template = await get_goal_template_from_db(id, request.sdb)
    try:
        await get_user_account_status_from_request(
            request, template[DBGoalTemplate.account_id.name],
        )
    except ResponseError:
        raise GoalTemplateNotFoundError(id) from None
    await update_goal_template_in_db(id, update_request.name, request.sdb)
    return web.json_response()
