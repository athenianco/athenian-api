from itertools import chain, groupby
from operator import itemgetter

from aiohttp import web

from athenian.api.align.goals.dbaccess import fetch_team_goals
from athenian.api.align.goals.measure import GoalToServe
from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.jira import get_jira_installation_or_none
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import Settings
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.team_metrics import calculate_team_metrics
from athenian.api.internal.team_tree import build_team_tree_from_rows
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Goal, Team
from athenian.api.models.web import AlignGoalsRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response


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
    logical_settings = await Settings.from_request(
        request, goals_request.account, prefixer,
    ).list_logical_repositories()

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
            logical_settings,
            jira_config,
            goals_request.only_with_targets,
            goals_request.include_series,
        )
        goals_to_serve.append(goal_to_serve)

    all_metric_values = await calculate_team_metrics(
        list(chain.from_iterable(g.requests for g in goals_to_serve)),
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

    models = [to_serve.build_goal_tree(all_metric_values) for to_serve in goals_to_serve]
    return model_response(models)
