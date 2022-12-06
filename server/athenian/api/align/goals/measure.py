"""Tools to build a Goals tree with measured metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from athenian.api.align.goals.dates import (
    GoalTimeseriesSpec,
    goal_datetimes_to_dates,
    goal_initial_query_interval,
)
from athenian.api.align.goals.dbaccess import (
    AliasedGoalColumns,
    GoalColumnAlias,
    TeamGoalColumns,
    convert_metric_params_datatypes,
    resolve_goal_repositories,
)
from athenian.api.db import Row
from athenian.api.internal.jira import JIRAConfig, check_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.internal.team_metrics import (
    CalcTeamMetricsRequest,
    MetricWithParams,
    RequestedTeamDetails,
    TeamMetricsResult,
)
from athenian.api.models.state.models import Goal, TeamGoal
from athenian.api.models.web.goal import (
    GoalMetricSeriesPoint,
    GoalTree,
    GoalValue,
    MetricValue,
    TeamGoalTree,
    TeamTree,
)
from athenian.api.tracing import sentry_span


class _GoalTreeGenerator:
    """Generate the response GoalTree for a goal."""

    @sentry_span
    def __call__(
        self,
        team_tree: TeamTree,
        goal_row: Row,
        team_goal_rows: Iterable[Row],
        metric_values: _GoalMetricValues,
        prefixer: Prefixer,
        logical_settings: LogicalRepositorySettings,
    ) -> GoalTree:
        """Compose the GoalTree for a goal from various piece of information."""
        valid_from, expires_at = goal_datetimes_to_dates(
            goal_row[Goal.valid_from.name], goal_row[Goal.expires_at.name],
        )
        team_goal_rows_map = {row[TeamGoal.team_id.name]: row for row in team_goal_rows}

        if (repos := goal_row[GoalColumnAlias.REPOSITORIES.value]) is not None:
            repos = [
                str(repo_name)
                for repo_name in resolve_goal_repositories(
                    repos, goal_row[Goal.id.name], prefixer, logical_settings,
                )
            ]

        team_goal = self._team_tree_to_team_goal_tree(team_tree, team_goal_rows_map, metric_values)
        return GoalTree(
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
            metric_params=goal_row[GoalColumnAlias.METRIC_PARAMS.value],
        )

    @classmethod
    def _team_tree_to_team_goal_tree(
        cls,
        team_tree: TeamTree,
        team_goal_rows_map: Mapping[int, Row],
        metric_values: _GoalMetricValues,
    ) -> TeamGoalTree:
        team_id = team_tree.id
        try:
            team_goal_row = team_goal_rows_map[team_id]
        except KeyError:
            # the team can be present in the tree but have no team goal
            target = metric_params = None
            goal_id = 0
        else:
            goal_id = team_goal_row[TeamGoal.goal_id.name]
            target = team_goal_row[TeamGoal.target.name]
            metric_params = team_goal_row[TeamGoal.metric_params.name]

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
            team_tree,
            initial,
            current,
            target,
            metric_params,
            series,
            metric_values.series_spec,
            children,
        )

    @classmethod
    def _compose_team_goal_tree(
        cls,
        team_tree: TeamTree,
        initial: MetricValue,
        current: MetricValue,
        target: MetricValue,
        metric_params: Optional[dict],
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
        return TeamGoalTree(
            team=team, value=goal_value, metric_params=metric_params, children=children,
        )


class GoalToServe:
    """A goal served in the response."""

    _goal_tree_generator = _GoalTreeGenerator()

    def __init__(
        self,
        team_goal_rows: Sequence[Row],
        team_tree: TeamTree,
        team_member_map: dict[int, list[int]],
        prefixer: Prefixer,
        logical_settings: LogicalRepositorySettings,
        jira_config: Optional[JIRAConfig],
        only_with_targets: bool,
        include_series: bool,
    ):
        """Init the GoalToServe object."""
        self._team_goal_rows = team_goal_rows
        self._prefixer = prefixer
        self._logical_settings = logical_settings
        self._parse_team_goal_rows(
            team_goal_rows,
            team_tree,
            team_member_map,
            prefixer,
            logical_settings,
            jira_config,
            only_with_targets,
            include_series,
        )

    @property
    def requests(self) -> list[CalcTeamMetricsRequest]:
        """Get the request for calculate_team_metrics to compute the metrics for the goal."""
        return self._requests

    def build_goal_tree(self, metric_values: TeamMetricsResult) -> GoalTree:
        """Build the GoalTree combining the teams structure and the metric values."""
        intervals = self._requests[0].time_intervals  # the same in each request

        initial_metrics_base = metric_values[intervals[0]]
        initial_metrics = {}
        current_metrics_base = metric_values[intervals[1]]
        current_metrics = {}
        if self._timeseries_spec is None:
            series: Optional[dict] = None
        else:
            series = {}
        # parse metric_values so that it can be more easily consumed by generator
        # for each team read the value for the metric with the right parameters
        for t_id_g_id, params in self._metrics_w_params_by_team.items():
            initial_metrics[t_id_g_id] = initial_metrics_base[params][t_id_g_id][0]
            current_metrics[t_id_g_id] = current_metrics_base[params][t_id_g_id][0]
            if series is not None and self._timeseries_spec is not None:
                intervals = self._timeseries_spec.intervals
                series[t_id_g_id] = metric_values[intervals][params][t_id_g_id]

        metric_values = _GoalMetricValues(
            initial_metrics, current_metrics, series, self._timeseries_spec,
        )

        return self._goal_tree_generator(
            self._goal_team_tree,
            self._team_goal_rows[0],
            self._team_goal_rows,
            metric_values,
            self._prefixer,
            self._logical_settings,
        )

    def _parse_team_goal_rows(
        self,
        team_goal_rows: Sequence[Row],
        team_tree: TeamTree,
        team_member_map: dict[int, list[int]],
        prefixer: Prefixer,
        logical_settings: LogicalRepositorySettings,
        unchecked_jira_config: Optional[JIRAConfig],
        only_with_targets: bool,
        include_series: bool,
    ) -> None:
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
        requests = []
        metrics_w_params_by_team = {}
        time_intervals = [initial_interval, (valid_from, expires_at)]
        if include_series:
            timeseries_spec = GoalTimeseriesSpec.from_timespan(valid_from, expires_at)
            time_intervals.append(timeseries_spec.intervals)
        else:
            timeseries_spec = None

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
                repo_names = resolve_goal_repositories(
                    repositories, goal_id, prefixer, logical_settings,
                )
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

            team_details = RequestedTeamDetails(
                team_id=team_id,
                goal_id=goal_id,
                members=team_member_map[team_id],
                repositories=repositories,
                jira_filter=jira_filter,
            )
            metric_params = convert_metric_params_datatypes(
                team_goal_row[columns[TeamGoal.metric_params.name]],
            )
            metric_w_params = MetricWithParams(metric, metric_params)
            metrics_w_params_by_team[(team_id, goal_id)] = metric_w_params
            # build a different request for every team
            # requests will be then simplified by calculate_team_metrics
            requests.append(
                CalcTeamMetricsRequest(
                    metrics=[metric_w_params],
                    time_intervals=time_intervals,
                    teams=[team_details],
                ),
            )

        self._requests = requests
        self._goal_team_tree = pruned_tree
        self._timeseries_spec = timeseries_spec
        self._metrics_w_params_by_team = metrics_w_params_by_team


@dataclass(slots=True, frozen=True)
class _GoalMetricValues:
    """The metric values for a Goal across all teams."""

    initial: Mapping[tuple[int, int], Any]
    current: Mapping[tuple[int, int], Any]
    series: Optional[Mapping[tuple[int, int], list[Any]]]
    series_spec: Optional[GoalTimeseriesSpec]


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
