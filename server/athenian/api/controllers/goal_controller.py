from itertools import groupby
from operator import itemgetter
from typing import Optional

from aiohttp import web

from athenian.api.align.exceptions import GoalTemplateNotFoundError
from athenian.api.align.goals.dates import goal_dates_to_datetimes
from athenian.api.align.goals.dbaccess import (
    GoalCreationInfo,
    delete_goal as db_delete_goal,
    delete_goal_template_from_db,
    dump_goal_repositories,
    fetch_goal_account,
    fetch_team_goals,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal,
    insert_goal_template,
    parse_goal_repositories,
    update_goal_template_in_db,
)
from athenian.api.align.goals.measure import GoalToServe, GoalTreeGenerator
from athenian.api.align.queries.metrics import calculate_team_metrics
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.db import Row, integrity_errors
from athenian.api.internal.account import (
    get_metadata_account_ids,
    get_user_account_status_from_request,
)
from athenian.api.internal.jira import (
    get_jira_installation_or_none,
    normalize_issue_type,
    normalize_priority,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.team_tree import build_team_tree_from_rows
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Goal, GoalTemplate as DBGoalTemplate, Team, TeamGoal
from athenian.api.models.web import (
    AlignGoalsRequest,
    BadRequestError,
    CreatedIdentifier,
    DatabaseConflict,
    GoalCreateRequest,
    GoalTemplate,
    GoalTemplateCreateRequest,
    GoalTemplateUpdateRequest,
    InvalidRequestError,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span


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

    repositories = await parse_request_repositories(
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
    return model_response(CreatedIdentifier(id=template_id))


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
    repositories = await parse_request_repositories(
        update_request.repositories, request, account_id,
    )
    values = {
        DBGoalTemplate.name.name: update_request.name,
        DBGoalTemplate.metric.name: update_request.metric,
        DBGoalTemplate.repositories.name: repositories,
    }
    await update_goal_template_in_db(id, request.sdb, **values)
    return web.json_response({})


async def parse_request_repositories(
    repo_names: Optional[list[str]],
    request: AthenianWebRequest,
    account_id: int,
) -> Optional[list[tuple[int, str]]]:
    """Resolve repository node IDs from the prefixed names."""
    if repo_names is None:
        return None
    prefixer = await Prefixer.from_request(request, account_id)
    try:
        return dump_goal_repositories(prefixer.prefixed_repo_names_to_identities(repo_names))
    except ValueError as e:
        raise ResponseError(InvalidRequestError(".repositories", str(e)))


@weight(10)
async def measure_goals(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate the metrics for the goal tree."""
    goals_request = AlignGoalsRequest.from_dict(body)
    team = goals_request.team
    team_rows, meta_ids, jira_config = await gather(
        fetch_teams_recursively(
            goals_request.account,
            request.sdb,
            select_entities=(Team.id, Team.name, Team.members, Team.parent_id),
            # teamId 0 means to implicitly use the single root team
            root_team_ids=None if team == 0 else [team],
        ),
        get_metadata_account_ids(goals_request.account, request.sdb, request.cache),
        get_jira_installation_or_none(
            goals_request.account, request.sdb, request.mdb, request.cache,
        ),
    )
    team_tree = build_team_tree_from_rows(team_rows, None if team == 0 else team)
    team_member_map = flatten_teams(team_rows)

    team_ids = [row[Team.id.name] for row in team_rows]
    team_goal_rows, prefixer = await gather(
        fetch_team_goals(goals_request.account, team_ids, request.sdb),
        Prefixer.load(meta_ids, request.mdb, request.cache),
    )

    goals_to_serve = []
    # iter all team goal rows, grouped by goal, to build GoalToServe object for the goal
    # fetch_team_goals result is ordered by Goal id so the groupby works as expected
    for _, group_team_goal_rows_iter in groupby(team_goal_rows, itemgetter(Goal.id.name)):
        goal_team_goal_rows = list(group_team_goal_rows_iter)
        goal_to_serve = GoalToServe(
            goal_team_goal_rows,
            team_tree,
            team_member_map,
            prefixer,
            jira_config,
            goals_request.only_with_targets,
            goals_request.include_series,
        )
        goals_to_serve.append(goal_to_serve)

    all_metric_values = await calculate_team_metrics(
        [g.request for g in goals_to_serve],
        account=goals_request.account,
        meta_ids=meta_ids,
        sdb=request.sdb,
        mdb=request.mdb,
        pdb=request.pdb,
        rdb=request.rdb,
        cache=request.cache,
        slack=request.app["slack"],
        unchecked_jira_config=jira_config,
    )

    goal_tree_generator = GoalTreeGenerator()
    models = [
        to_serve.build_goal_tree(all_metric_values, goal_tree_generator)
        for to_serve in goals_to_serve
    ]
    return model_response(models)


@disable_default_user
async def delete_goal(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete an existing goal."""
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            account = await fetch_goal_account(id, sdb_conn)
            await get_user_account_status_from_request(request, account)
            await db_delete_goal(account, id, sdb_conn)

    return web.Response(status=204)


@disable_default_user
async def create_goal(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a new goal."""
    try:
        create_request = GoalCreateRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e

    creation_info = await _parse_create_request(request, create_request)
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            new_goal_id = await insert_goal(creation_info, sdb_conn)

    return model_response(CreatedIdentifier(id=new_goal_id))


@sentry_span
async def _parse_create_request(
    request: AthenianWebRequest,
    creat_req: GoalCreateRequest,
) -> GoalCreationInfo:
    valid_from, expires_at = goal_dates_to_datetimes(creat_req.valid_from, creat_req.expires_at)
    if expires_at < valid_from:
        raise ResponseError(BadRequestError(detail="Goal expires_at cannot precede valid_from"))

    repositories = await parse_request_repositories(
        creat_req.repositories, request, creat_req.account,
    )
    jira_projects = creat_req.jira_projects
    if creat_req.jira_priorities is None:
        jira_priorities = None
    else:
        jira_priorities = sorted({normalize_priority(p) for p in creat_req.jira_priorities})
    if creat_req.jira_issue_types is None:
        jira_issue_types = None
    else:
        jira_issue_types = sorted({normalize_issue_type(t) for t in creat_req.jira_issue_types})

    # user cannot directly set TeamGoal filter fields, received goal values are applied
    extra_team_goal_info = {
        TeamGoal.repositories.name: repositories,
        TeamGoal.jira_projects.name: jira_projects,
        TeamGoal.jira_priorities.name: jira_priorities,
        TeamGoal.jira_issue_types.name: jira_issue_types,
    }
    team_goals = [
        TeamGoal(team_id=tg.team_id, target=tg.target, **extra_team_goal_info)
        for tg in creat_req.team_goals
    ]

    if len({team_goal.team_id for team_goal in team_goals}) < len(team_goals):
        raise ResponseError(BadRequestError("More than one team goal with the same teamId"))
    goal = Goal(
        account_id=creat_req.account,
        name=creat_req.name,
        metric=creat_req.metric,
        repositories=repositories,
        jira_projects=jira_projects,
        jira_priorities=jira_priorities,
        jira_issue_types=jira_issue_types,
        valid_from=valid_from,
        expires_at=expires_at,
    )
    return GoalCreationInfo(goal, team_goals)
