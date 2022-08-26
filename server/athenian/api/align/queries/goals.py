from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ariadne import ObjectType
from graphql import GraphQLResolveInfo

from athenian.api.align.goals.dates import goal_datetimes_to_dates, goal_initial_query_interval
from athenian.api.align.goals.dbaccess import fetch_team_goals, resolve_goal_repositories
from athenian.api.align.models import GoalTree, GoalValue, MetricValue, TeamGoalTree, TeamTree
from athenian.api.align.queries.metrics import (
    RequestedTeamDetails,
    TeamMetricsRequest,
    calculate_team_metrics,
)
from athenian.api.align.queries.teams import build_team_tree_from_rows
from athenian.api.async_utils import gather
from athenian.api.db import Row
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Goal, Team, TeamGoal
from athenian.api.tracing import sentry_span

query = ObjectType("Query")


@query.field("goals")
@sentry_span
async def resolve_goals(
    obj: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    teamId: int,
    onlyWithTargets: bool,
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
    team_goal_rows, prefixer = await gather(
        fetch_team_goals(accountId, team_ids, info.context.sdb),
        Prefixer.load(meta_ids, info.context.mdb, info.context.cache),
    )

    goals_to_serve = []

    # iter all team goal rows, grouped by goal, to build _GoalToServe object for the goal
    for _, group_team_goal_rows_iter in groupby(team_goal_rows, itemgetter(Goal.id.name)):
        goal_team_goal_rows = list(group_team_goal_rows_iter)
        goals_to_serve.append(
            _GoalToServe(
                goal_team_goal_rows, team_tree, team_member_map, prefixer, onlyWithTargets,
            ),
        )

    all_metric_values = await calculate_team_metrics(
        [g.request for g in goals_to_serve],
        account=accountId,
        meta_ids=meta_ids,
        sdb=info.context.sdb,
        mdb=info.context.mdb,
        pdb=info.context.pdb,
        rdb=info.context.rdb,
        cache=info.context.cache,
        slack=info.context.app["slack"],
    )
    return [to_serve.build_goal_tree(all_metric_values).to_dict() for to_serve in goals_to_serve]


class _GoalToServe:
    """A goal served in the response."""

    def __init__(
        self,
        team_goal_rows: Sequence[Row],
        team_tree: TeamTree,
        team_member_map: dict[int, list[int]],
        prefixer: Prefixer,
        only_with_targets: bool,
    ):
        self._team_goal_rows = team_goal_rows
        self._request, self._goal_team_tree = self._team_goal_rows_to_request(
            team_goal_rows, team_tree, team_member_map, prefixer, only_with_targets,
        )

    @property
    def request(self) -> TeamMetricsRequest:
        return self._request

    def build_goal_tree(self, metric_values) -> GoalTree:
        initial_metrics = metric_values[self._request.time_intervals[0]][self._request.metrics[0]]
        current_metrics = metric_values[self._request.time_intervals[1]][self._request.metrics[0]]
        metric_values = GoalMetricValues(initial_metrics, current_metrics)
        return _team_tree_to_goal_tree(
            self._goal_team_tree, self._team_goal_rows[0], self._team_goal_rows, metric_values,
        )

    @classmethod
    def _team_goal_rows_to_request(
        cls,
        team_goal_rows: Sequence[Row],
        team_tree: TeamTree,
        team_member_map: dict[int, list[int]],
        prefixer: Prefixer,
        only_with_targets: bool,
    ) -> tuple[TeamMetricsRequest, TeamTree]:
        goal_row = team_goal_rows[0]  # could be any, all rows have the joined Goal columns
        metric = goal_row[Goal.metric.name]

        valid_from = goal_row[Goal.valid_from.name]
        expires_at = goal_row[Goal.expires_at.name]
        initial_interval = goal_initial_query_interval(valid_from, expires_at)

        teams_map = {row[TeamGoal.team_id.name]: row for row in team_goal_rows}

        # FIXME: why teams_map misses some IDs from team_member_map?

        if only_with_targets:
            # 1 - prune team tree to keep only branches with assigned teams
            goal_team_tree = _team_tree_prune_empty_branches(team_tree, lambda id: id in teams_map)
            # assigned_teams is a non empty subset of all the ids in team_tree,
            # so the pruned tree is never empty
            assert goal_team_tree is not None
            # 2 - in the metrics requests only ask for teams present in the pruned tree
            goal_tree_team_ids = goal_team_tree.flatten_team_ids()
            requested_teams = {
                team_id: RequestedTeamDetails(
                    members,
                    resolve_goal_repositories(
                        teams_map[team_id][TeamGoal.repositories.name], prefixer,
                    ),
                )
                for team_id, members in team_member_map.items()
                if team_id in goal_tree_team_ids
                and team_id in teams_map  # FIXME: why is this needed??
            }
        else:
            goal_team_tree = team_tree
            requested_teams = {
                team_id: RequestedTeamDetails(
                    members,
                    resolve_goal_repositories(
                        teams_map[team_id][TeamGoal.repositories.name], prefixer,
                    ),
                )
                for team_id, members in team_member_map.items()
                if team_id in teams_map  # FIXME: why is this needed??
            }

        team_metrics_request = TeamMetricsRequest(
            metrics=[metric],
            time_intervals=(initial_interval, (valid_from, expires_at)),
            teams=requested_teams,
        )
        return team_metrics_request, goal_team_tree


@dataclass(slots=True)
class GoalMetricValues:
    """The metric values for a Goal across all teams."""

    initial: Mapping[int, Any]
    current: Mapping[int, Any]


@sentry_span
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
        goal_row[Goal.name.name],
        goal_row[Goal.metric.name],
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
        target = None
    else:
        target = MetricValue(team_goal_row[TeamGoal.target.name])

    goal_value = GoalValue(
        current=MetricValue(metric_values.current.get(team_tree.id)),
        initial=MetricValue(metric_values.initial.get(team_tree.id)),
        target=target,
    )
    children = [
        _team_tree_to_team_goal_tree(child, team_goal_rows_map, metric_values)
        for child in team_tree.children
    ]
    return TeamGoalTree(team_tree, goal_value, children)


@sentry_span
def _team_tree_prune_empty_branches(
    team_tree: TeamTree,
    keep_team_fn: Callable[[int], bool],
) -> Optional[TeamTree]:
    """Remove unwanted teams from a TeamTree.

    Empty branches are pruned.
    """
    kept_children = [
        pruned_child
        for c in team_tree.children
        if (pruned_child := _team_tree_prune_empty_branches(c, keep_team_fn)) is not None
    ]
    if kept_children or keep_team_fn(team_tree.id):
        return team_tree.with_children(kept_children)
    else:
        return None
