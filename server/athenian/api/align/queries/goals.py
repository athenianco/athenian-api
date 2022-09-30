from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import Any, Callable, Generic, Iterable, Mapping, Optional, Sequence, Type, TypeVar

from ariadne import ObjectType
from graphql import GraphQLResolveInfo

from athenian.api.align.goals.dates import (
    GoalTimeseriesSpec,
    goal_datetimes_to_dates,
    goal_initial_query_interval,
)
from athenian.api.align.goals.dbaccess import (
    AliasedGoalColumns,
    GoalColumnAlias,
    TeamGoalColumns,
    fetch_team_goals,
    resolve_goal_repositories,
)
from athenian.api.align.models import (
    GraphQLGoalTree,
    GraphQLGoalValue,
    GraphQLMetricValue,
    GraphQLTeamGoalTree,
    GraphQLTeamTree,
)
from athenian.api.align.queries.metrics import (
    RequestedTeamDetails,
    TeamMetricsRequest,
    TeamMetricsResult,
    calculate_team_metrics,
)
from athenian.api.align.queries.teams import build_team_tree_from_rows
from athenian.api.async_utils import gather
from athenian.api.db import Row
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.jira import (
    JIRAConfig,
    check_jira_installation,
    get_jira_installation_or_none,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Goal, Team, TeamGoal
from athenian.api.models.web.goal import (
    GoalMetricSeriesPoint,
    GoalTree,
    GoalValue,
    MetricValue,
    TeamGoalTree,
    TeamTree,
)
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
    team_rows, meta_ids, jira_config = await gather(
        fetch_teams_recursively(
            accountId,
            info.context.sdb,
            select_entities=(Team.id, Team.name, Team.members, Team.parent_id),
            # teamId 0 means to implicitly use the single root team
            root_team_ids=None if teamId == 0 else [teamId],
        ),
        get_metadata_account_ids(accountId, info.context.sdb, info.context.cache),
        get_jira_installation_or_none(
            accountId, info.context.sdb, info.context.mdb, info.context.cache,
        ),
    )
    team_tree = build_team_tree_from_rows(team_rows, None if teamId == 0 else teamId)
    team_member_map = flatten_teams(team_rows)
    team_ids = [row[Team.id.name] for row in team_rows]
    team_goal_rows, prefixer = await gather(
        fetch_team_goals(accountId, team_ids, info.context.sdb),
        Prefixer.load(meta_ids, info.context.mdb, info.context.cache),
    )

    goals_to_serve = []

    # iter all team goal rows, grouped by goal, to build GoalToServe object for the goal
    for _, group_team_goal_rows_iter in groupby(team_goal_rows, itemgetter(Goal.id.name)):
        goal_team_goal_rows = list(group_team_goal_rows_iter)
        goal_to_serve = GoalToServe(
            goal_team_goal_rows,
            team_tree,
            team_member_map,
            prefixer,
            jira_config,
            onlyWithTargets,
            False,
        )
        goals_to_serve.append(goal_to_serve)

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
        unchecked_jira_config=jira_config,
    )
    goal_tree_generator = GraphQLGoalTreeGenerator()
    return [
        to_serve.build_goal_tree(all_metric_values, goal_tree_generator).to_dict()
        for to_serve in goals_to_serve
    ]


class GoalToServe:
    """A goal served in the response."""

    def __init__(
        self,
        team_goal_rows: Sequence[Row],
        team_tree: TeamTree,
        team_member_map: dict[int, list[int]],
        prefixer: Prefixer,
        jira_config: Optional[JIRAConfig],
        only_with_targets: bool,
        include_series: bool,
    ):
        """Init the GoalToServe object."""
        self._team_goal_rows = team_goal_rows
        self._prefixer = prefixer
        self._request, self._goal_team_tree, self._timeseries_spec = self._parse_team_goal_rows(
            team_goal_rows,
            team_tree,
            team_member_map,
            prefixer,
            jira_config,
            only_with_targets,
            include_series,
        )

    @property
    def request(self) -> TeamMetricsRequest:
        """Get the request for calculate_team_metrics to compute the metrics for the goal."""
        return self._request

    def build_goal_tree(
        self,
        metric_values: TeamMetricsResult,
        generator: GoalTreeGenerator,
    ) -> GoalTree:
        """Build the GoalTree combining the teams structure and the metric values."""
        intervals = self._request.time_intervals
        metric = self._request.metrics[0]

        # for initial and current metrics the interval is a pair,
        # we produce and need only one value from TeamMetricsResult
        initial_metrics_base = metric_values[intervals[0]][metric]
        initial_metrics = {k: values[0] for k, values in initial_metrics_base.items()}
        current_metrics_values = metric_values[intervals[1]][metric]
        current_metrics = {k: values[0] for k, values in current_metrics_values.items()}

        if self._timeseries_spec is None:
            series = None
        else:
            series = metric_values[self._timeseries_spec.intervals][metric]

        metric_values = GoalMetricValues(
            initial_metrics, current_metrics, series, self._timeseries_spec,
        )

        return generator(
            self._goal_team_tree,
            self._team_goal_rows[0],
            self._team_goal_rows,
            metric_values,
            self._prefixer,
        )

    @classmethod
    def _parse_team_goal_rows(
        cls,
        team_goal_rows: Sequence[Row],
        team_tree: TeamTree,
        team_member_map: dict[int, list[int]],
        prefixer: Prefixer,
        unchecked_jira_config: Optional[JIRAConfig],
        only_with_targets: bool,
        include_series: bool,
    ) -> tuple[TeamMetricsRequest, TeamTree, Optional[GoalTimeseriesSpec]]:
        goal_row = team_goal_rows[0]  # could be any, all rows have the joined Goal columns
        metric = goal_row[Goal.metric.name]

        valid_from = goal_row[Goal.valid_from.name]
        expires_at = goal_row[Goal.expires_at.name]
        initial_interval = goal_initial_query_interval(valid_from, expires_at)

        rows_by_team = {row[TeamGoal.team_id.name]: row for row in team_goal_rows}
        if only_with_targets:
            # prune team tree to keep only branches with assigned teams
            pruned_tree = _team_tree_prune_empty_branches(team_tree, lambda id: id in rows_by_team)
            # id in rows_by_team is a non empty subset of all the ids in team_tree,
            # so the pruned tree is never empty
            assert pruned_tree is not None
        else:
            pruned_tree = team_tree

        # in the metrics requests only ask for teams present in the pruned tree
        goal_tree_team_ids = pruned_tree.flatten_team_ids()
        requested_teams = []
        for team_id in goal_tree_team_ids:
            try:
                team_goal_row = rows_by_team[team_id]
            except KeyError:
                team_goal_row = goal_row
                columns = AliasedGoalColumns
                goal_id = 0  # the filters are shared
            else:
                columns = TeamGoalColumns
                goal_id = team_goal_row[TeamGoal.goal_id.name]
            repositories = team_goal_row[columns[TeamGoal.repositories.name]]
            if repositories is not None:
                repo_names = resolve_goal_repositories(repositories, prefixer)
                repositories = tuple(name.unprefixed for name in repo_names)

            jira_projects = team_goal_row[columns[TeamGoal.jira_projects.name]]
            jira_priorities = team_goal_row[columns[TeamGoal.jira_priorities.name]]
            jira_issue_types = team_goal_row[columns[TeamGoal.jira_issue_types.name]]

            if jira_projects or jira_priorities or jira_issue_types:
                jira_config = check_jira_installation(unchecked_jira_config)
                if jira_projects:
                    jira_projects = frozenset(jira_config.translate_project_keys(jira_projects))
                    custom_projects = True
                else:
                    jira_projects = frozenset(jira_config.projects)
                    custom_projects = False
                jira_filter = JIRAFilter(
                    jira_config.acc_id,
                    jira_projects,
                    LabelFilter.empty(),
                    frozenset(),
                    frozenset(jira_issue_types or ()),
                    frozenset(jira_priorities or ()),
                    custom_projects,
                    False,
                )
            else:
                jira_filter = JIRAFilter.empty()

            requested_teams.append(
                RequestedTeamDetails(
                    team_id=team_id,
                    goal_id=goal_id,
                    members=team_member_map[team_id],
                    repositories=repositories,
                    jira_filter=jira_filter,
                ),
            )

        time_intervals = [initial_interval, (valid_from, expires_at)]
        if include_series:
            timeseries_spec = GoalTimeseriesSpec.from_timespan(valid_from, expires_at)
            time_intervals.append(timeseries_spec.intervals)
        else:
            timeseries_spec = None

        team_metrics_request = TeamMetricsRequest(
            metrics=[metric], time_intervals=time_intervals, teams=requested_teams,
        )
        return team_metrics_request, pruned_tree, timeseries_spec


@dataclass(slots=True, frozen=True)
class GoalMetricValues:
    """The metric values for a Goal across all teams."""

    initial: Mapping[tuple[int, int], Any]
    current: Mapping[tuple[int, int], Any]
    series: Optional[Mapping[tuple[int, int], list[Any]]]
    series_spec: Optional[GoalTimeseriesSpec]


GoalTreeType = TypeVar("GoalTreeType", bound=GoalTree)


class GoalTreeGenerator(Generic[GoalTreeType]):
    """Generate the response GoalTree for a goal."""

    goal_tree_class: Type[GoalTreeType] = GoalTree

    @sentry_span
    def __call__(
        self,
        team_tree: TeamTree,
        goal_row: Row,
        team_goal_rows: Iterable[Row],
        metric_values: GoalMetricValues,
        prefixer: Prefixer,
    ) -> GoalTreeType:
        """Compose the GoalTree for a goal from various piece of information."""
        valid_from, expires_at = goal_datetimes_to_dates(
            goal_row[Goal.valid_from.name], goal_row[Goal.expires_at.name],
        )
        team_goal_rows_map = {row[TeamGoal.team_id.name]: row for row in team_goal_rows}

        if (repos := goal_row[GoalColumnAlias.REPOSITORIES.value]) is not None:
            repos = [str(repo_name) for repo_name in resolve_goal_repositories(repos, prefixer)]

        team_goal = self._team_tree_to_team_goal_tree(team_tree, team_goal_rows_map, metric_values)
        return self.goal_tree_class(
            id=goal_row[Goal.id.name],
            name=goal_row[Goal.name.name],
            metric=goal_row[Goal.metric.name],
            valid_from=valid_from,
            expires_at=expires_at,
            team_goal=team_goal,
            repositories=repos,
            jira_projects=goal_row[GoalColumnAlias.JIRA_PROJECTS.value],
            jira_priorities=goal_row[GoalColumnAlias.JIRA_PRIORITIES.value],
            jira_issue_types=goal_row[GoalColumnAlias.JIRA_ISSUE_TYPES.value],
        )

    @classmethod
    def _team_tree_to_team_goal_tree(
        cls,
        team_tree: TeamTree,
        team_goal_rows_map: Mapping[int, Row],
        metric_values: GoalMetricValues,
    ) -> TeamGoalTree:
        team_id = team_tree.id
        try:
            team_goal_row = team_goal_rows_map[team_id]
        except KeyError:
            # the team can be present in the tree but have no team goal
            target = None
            goal_id = 0
        else:
            goal_id = team_goal_row[TeamGoal.goal_id.name]
            target = team_goal_row[TeamGoal.target.name]

        children = [
            cls._team_tree_to_team_goal_tree(child, team_goal_rows_map, metric_values)
            for child in team_tree.children
        ]

        current = metric_values.current.get((team_id, goal_id))
        initial = metric_values.initial.get((team_id, goal_id))
        if metric_values.series is None:
            series = None
        else:
            series = metric_values.series.get((team_id, goal_id))

        return cls._compose_team_goal_tree(
            team_tree, initial, current, target, series, metric_values.series_spec, children,
        )

    @classmethod
    def _compose_team_goal_tree(
        cls,
        team_tree: TeamTree,
        initial: MetricValue,
        current: MetricValue,
        target: MetricValue,
        series: Optional[list[MetricValue]],
        series_spec: Optional[GoalTimeseriesSpec],
        children: Sequence[TeamGoalTree],
    ) -> TeamGoalTree:
        team = team_tree.as_leaf()
        if series is None or series_spec is None:
            series = None
            series_granularity = None
        else:
            series_dates = [d.date() for d in series_spec.intervals[:-1]]
            series = [GoalMetricSeriesPoint(date=d, value=v) for d, v in zip(series_dates, series)]
            series_granularity = series_spec.granularity

        goal_value = GoalValue(
            current=current,
            initial=initial,
            target=target,
            series=series,
            series_granularity=series_granularity,
        )
        return TeamGoalTree(team=team, value=goal_value, children=children)


class GraphQLGoalTreeGenerator(GoalTreeGenerator[GraphQLTeamGoalTree]):
    """Generate the response GoalTree for a goal."""

    goal_tree_class = GraphQLGoalTree

    @classmethod
    def _compose_team_goal_tree(
        cls,
        team_tree: TeamTree,
        initial: MetricValue,
        current: MetricValue,
        target: MetricValue,
        series: Optional[list[MetricValue]],
        series_spec: Optional[GoalTimeseriesSpec],
        children: Sequence[TeamGoalTree],
    ) -> GraphQLTeamGoalTree:
        team = GraphQLTeamTree.from_team_tree(team_tree)
        tgt = None if target is None else GraphQLMetricValue(target)
        goal_value = GraphQLGoalValue(
            current=GraphQLMetricValue(current), initial=GraphQLMetricValue(initial), target=tgt,
        )
        return GraphQLTeamGoalTree(team=team, value=goal_value, children=children)


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
