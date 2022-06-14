from itertools import chain, groupby
from operator import itemgetter
from typing import Any, Iterable, Mapping

from ariadne import ObjectType
from graphql import GraphQLResolveInfo

from athenian.api.align.goals.dates import goal_datetimes_to_dates, goal_initial_query_interval
from athenian.api.align.goals.dbaccess import fetch_team_goals
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.align.models import GoalTree, GoalValue, MetricValue, TeamGoalTree, TeamTree
from athenian.api.align.queries.metrics import calculate_team_metrics
from athenian.api.align.queries.teams import build_team_tree_from_rows
from athenian.api.async_utils import gather
from athenian.api.db import Row
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Goal, Team, TeamGoal
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass

query = ObjectType("Query")


@query.field("goals")
@sentry_span
async def resolve_goals(
    obj: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    teamId: int,
    **kwargs,
) -> Any:
    """Serve goals() query."""
    team_rows, meta_ids = await gather(
        fetch_teams_recursively(
            accountId,
            info.context.sdb,
            select_entities=(Team.id, Team.name, Team.members, Team.parent_id),
            # teamId 0 means to implicitly use the single root team
            root_team_ids=None if teamId == 0 else [teamId],
        ),
        get_metadata_account_ids(accountId, info.context.sdb, info.context.cache),
    )

    team_tree = build_team_tree_from_rows(team_rows, None if teamId == 0 else teamId)
    team_member_map = flatten_teams(team_rows)
    team_ids = [row[Team.id.name] for row in team_rows]
    team_goal_rows = await fetch_team_goals(accountId, team_ids, info.context.sdb)

    # first iter all team goal rows, grouped by goal, to collect all parameters for the single
    # metric request, and track what the goal requested in `goal_requests`
    goal_requests = []
    for _, group_team_goal_rows_iter in groupby(team_goal_rows, itemgetter(Goal.id.name)):
        group_team_goal_rows = list(group_team_goal_rows_iter)
        group_teams = {
            row[TeamGoal.team_id.name]: team_member_map[row[TeamGoal.team_id.name]]
            for row in group_team_goal_rows
        }
        group_goal_row = group_team_goal_rows[0]  # could be any row

        metric = TEMPLATES_COLLECTION[group_goal_row[Goal.template_id.name]]["metric"]

        valid_from = group_goal_row[Goal.valid_from.name]
        expires_at = group_goal_row[Goal.expires_at.name]
        initial_from, initial_to = goal_initial_query_interval(valid_from, expires_at)
        initial_interval = goal_initial_query_interval(valid_from, expires_at)

        goal_requests.append(
            {
                "metric": metric,
                "time_intervals": (initial_interval, (valid_from, expires_at)),
                "team_goal_rows": group_team_goal_rows,
                "teams": group_teams,
            }
        )

    all_metric_values = await calculate_team_metrics(
        metrics=list({r["metric"] for r in goal_requests}),
        time_intervals=list(set(chain.from_iterable(r["time_intervals"] for r in goal_requests))),
        teams=dict(chain.from_iterable(r["teams"].items() for r in goal_requests)),
        account=accountId,
        meta_ids=meta_ids,
        sdb=info.context.sdb,
        mdb=info.context.mdb,
        pdb=info.context.pdb,
        rdb=info.context.rdb,
        cache=info.context.cache,
        slack=info.context.app["slack"],
    )

    res = []
    for req in goal_requests:
        initial_metrics = all_metric_values[req["time_intervals"][0]][req["metric"]]
        current_metrics = all_metric_values[req["time_intervals"][1]][req["metric"]]

        metric_values = GoalMetricValues(initial_metrics, current_metrics)
        group_res = _team_tree_to_goal_tree(
            team_tree, req["team_goal_rows"][0], req["team_goal_rows"], metric_values,
        )
        res.append(group_res)

    return [r.to_dict() for r in res]


@dataclass(slots=True)
class GoalMetricValues:
    """The metric values for a Goal across all teams."""

    initial: Mapping[int, Any]
    current: Mapping[int, Any]


def _team_tree_to_goal_tree(
    team_tree: TeamTree,
    goal_row: Row,
    team_goal_rows: Iterable[Row],
    metric_values: GoalMetricValues,
) -> GoalTree:
    valid_from, expires_at = goal_datetimes_to_dates(
        goal_row[Goal.valid_from.name], goal_row[Goal.expires_at.name],
    )
    team_goal_rows_map = {row[TeamGoal.team_id.name]: row for row in team_goal_rows}
    return GoalTree(
        goal_row[Goal.id.name],
        goal_row[Goal.template_id.name],
        valid_from,
        expires_at,
        _team_tree_to_team_goal_tree(team_tree, team_goal_rows_map, metric_values),
    )


def _team_tree_to_team_goal_tree(
    team_tree: TeamTree,
    team_goal_rows_map: Mapping[int, Row],
    metric_values: GoalMetricValues,
) -> TeamGoalTree:
    try:
        team_goal_row = team_goal_rows_map[team_tree.id]
    except KeyError:
        # the team can be present in the tree but have no team goal
        goal_value = None
    else:
        goal_value = GoalValue(
            current=MetricValue(metric_values.current.get(team_tree.id)),
            initial=MetricValue(metric_values.initial.get(team_tree.id)),
            target=MetricValue(team_goal_row[TeamGoal.target.name]),
        )
    children = [
        _team_tree_to_team_goal_tree(child, team_goal_rows_map, metric_values)
        for child in team_tree.children
    ]
    return TeamGoalTree(team_tree, goal_value, children)
