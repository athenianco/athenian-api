from operator import attrgetter

from aiohttp import web

from athenian.api.align.goals.dbaccess import get_goal_template_from_db
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.internal.account import get_user_account_status_from_request
from athenian.api.models.state.models import GoalTemplate as DBGoalTemplate
from athenian.api.models.web import GoalTemplate
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response


async def get_goal_template(request: AthenianWebRequest, id: int) -> web.Response:
    """Retrieve a goal template.

    :param id: Numeric identifier of the goal template.
    :type id: int

    """
    row = await get_goal_template_from_db(id, request.sdb)
    await get_user_account_status_from_request(request, row[DBGoalTemplate.account_id.name])
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
    models = [
        GoalTemplate(id=id_, name=template["name"], metric=template["metric"])
        for id_, template in TEMPLATES_COLLECTION.items()
    ]
    models.sort(key=attrgetter("id"))
    return model_response(models)
