from aiohttp import web

from athenian.api.align.exceptions import GoalTemplateNotFoundError
from athenian.api.align.goals.dbaccess import (
    delete_goal_template_from_db,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal_template,
    update_goal_template_in_db,
)
from athenian.api.async_utils import gather
from athenian.api.db import Row, integrity_errors
from athenian.api.internal.account import (
    get_user_account_status_from_request,
    request_user_belongs_to_account,
)
from athenian.api.internal.datasources import AccountDatasources
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.repos import parse_db_repositories, parse_request_repositories
from athenian.api.models.state.models import GoalTemplate as DBGoalTemplate
from athenian.api.models.web import (
    CreatedIdentifier,
    DatabaseConflict,
    GoalTemplate,
    GoalTemplateCreateRequest,
    GoalTemplateUpdateRequest,
    JIRAMetricID,
)
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


async def get_goal_template(request: AthenianWebRequest, id: int) -> web.Response:
    """Retrieve a goal template.

    :param id: Numeric identifier of the goal template.
    :type id: int
    """
    row = await get_goal_template_from_db(id, request.sdb)
    account = row[DBGoalTemplate.account_id.name]
    if not await request_user_belongs_to_account(request, account):
        # do not leak the account that owns this template
        raise GoalTemplateNotFoundError(id)
    if (db_repos := parse_db_repositories(row[DBGoalTemplate.repositories.name])) is not None:
        prefixer = await Prefixer.from_request(request, account)
        repositories = [str(r) for r in prefixer.dereference_repositories(db_repos)]
    else:
        repositories = None
    datasources = await AccountDatasources.build_for_account(account, request.sdb)
    model = _goal_template_from_row(row, datasources, repositories=repositories)
    return model_response(model)


async def list_goal_templates(
    request: AthenianWebRequest,
    id: int,
    include_tlo: bool,
) -> web.Response:
    """List the goal templates for the account.

    :param id: Numeric identifier of the account.
    :type id: int
    """
    await get_user_account_status_from_request(request, id)
    rows, prefixer, datasources = await gather(
        get_goal_templates_from_db(id, not include_tlo, request.sdb),
        Prefixer.from_request(request, id),
        AccountDatasources.build_for_account(id, request.sdb),
    )

    models = []
    for row in rows:
        raw_db_repos = row[DBGoalTemplate.repositories.name]
        if (db_repos := parse_db_repositories(raw_db_repos)) is not None:
            repositories = [str(r) for r in prefixer.dereference_repositories(db_repos)]
        else:
            repositories = None
        models.append(_goal_template_from_row(row, datasources, repositories=repositories))
    return model_response(models)


async def create_goal_template(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a goal template.

    :param body: GoalTemplateCreateRequest
    """
    create_request = GoalTemplateCreateRequest.from_dict(body)
    await get_user_account_status_from_request(request, create_request.account)

    repositories = await parse_request_repositories(
        create_request.repositories, request, create_request.account,
    )
    values = {
        DBGoalTemplate.account_id.name: create_request.account,
        DBGoalTemplate.name.name: create_request.name,
        DBGoalTemplate.metric.name: create_request.metric,
        DBGoalTemplate.repositories.name: repositories,
        DBGoalTemplate.metric_params.name: create_request.metric_params,
    }
    try:
        template_id = await insert_goal_template(request.sdb, **values)
    except integrity_errors:
        raise ResponseError(
            DatabaseConflict(
                detail=f"Goal template named '{create_request.name}' already exists.",
            ),
        ) from None
    return model_response(CreatedIdentifier(id=template_id))


async def delete_goal_template(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a goal tamplate.

    :param id: Numeric identifier of the goal template.
    """
    template = await get_goal_template_from_db(id, request.sdb)
    account = template[DBGoalTemplate.account_id.name]
    if not await request_user_belongs_to_account(request, account):
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
    if not await request_user_belongs_to_account(request, account_id):
        raise GoalTemplateNotFoundError(id) from None
    repositories = await parse_request_repositories(
        update_request.repositories, request, account_id,
    )
    values = {
        DBGoalTemplate.name.name: update_request.name,
        DBGoalTemplate.metric.name: update_request.metric,
        DBGoalTemplate.repositories.name: repositories,
        DBGoalTemplate.metric_params.name: update_request.metric_params,
    }
    await update_goal_template_in_db(id, request.sdb, **values)
    return web.json_response({})


_JIRA_RELATED_METRICS = frozenset(JIRAMetricID) | {
    PullRequestMetricID.PR_OPENED_MAPPED_TO_JIRA,
    PullRequestMetricID.PR_DONE_MAPPED_TO_JIRA,
    PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
}


def _goal_template_from_row(row: Row, datasources: AccountDatasources, **kwargs) -> GoalTemplate:
    metric = row[DBGoalTemplate.metric.name]
    available = (metric not in _JIRA_RELATED_METRICS) or AccountDatasources.JIRA in datasources
    return GoalTemplate(
        available=available,
        id=row[DBGoalTemplate.id.name],
        name=row[DBGoalTemplate.name.name],
        metric=metric,
        metric_params=row[DBGoalTemplate.metric_params.name],
        **kwargs,
    )
