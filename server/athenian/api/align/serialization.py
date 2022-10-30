from typing import Optional


def parse_metric_params(metric_params: Optional[dict]) -> Optional[dict]:
    """Convert the raw input `GoalMetricParamsInput`."""
    if metric_params is None:
        return metric_params
    # unwrap the threshold param value if present
    parsed = metric_params.copy()
    try:
        parsed["threshold"] = parse_union_value(parsed["threshold"])
    except KeyError:
        pass
    return parsed


def parse_union_value(team_goal_target: dict) -> int | float | str:
    """Get the first non null value in GoalTargetInput."""
    return next(tgt for tgt in team_goal_target.values() if tgt is not None)
