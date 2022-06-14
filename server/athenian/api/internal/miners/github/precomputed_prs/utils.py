from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from dateutil.rrule import DAILY, rrule
from sqlalchemy import not_, or_
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.filters import LabelFilter
from athenian.api.internal.miners.types import (
    MinedPullRequest,
    PullRequestFacts,
    PullRequestFactsMap,
)
from athenian.api.internal.settings import ReleaseMatch, ReleaseSettings, default_branch_alias
from athenian.api.models.metadata.github import (
    PullRequestComment,
    PullRequestCommit,
    PullRequestReview,
    PullRequestReviewRequest,
)
from athenian.api.models.precomputed.models import GitHubBase


def build_days_range(time_from: datetime, time_to: datetime) -> Set[datetime]:
    """Build the daily range between the two provided times."""
    # timezones: date_from and date_to may be not exactly 00:00
    date_from_day = datetime.combine(time_from.date(), datetime.min.time(), tzinfo=timezone.utc)
    date_to_day = datetime.combine(time_to.date(), datetime.min.time(), tzinfo=timezone.utc)
    # date_to_day will be included
    return rrule(DAILY, dtstart=date_from_day, until=date_to_day)


def append_activity_days_filter(
    time_from: datetime,
    time_to: datetime,
    selected: Set[InstrumentedAttribute],
    filters: List[ClauseElement],
    activity_days_column: InstrumentedAttribute,
    postgres: bool,
) -> Set[datetime]:
    """Append the activity days to provided SQL filters."""
    date_range = build_days_range(time_from, time_to)
    if postgres:
        filters.append(activity_days_column.overlap(list(date_range)))
    else:
        selected.add(activity_days_column)
        date_range = set(date_range)
    return date_range


def collect_activity_days(pr: MinedPullRequest, facts: PullRequestFacts, sqlite: bool):
    """Collect activity days from mined PR and facts."""
    activity_days = set()
    if facts.released is not None:
        activity_days.add(facts.released.item().date())
    if facts.closed is not None:
        activity_days.add(facts.closed.item().date())
    activity_days.add(facts.created.item().date())
    # if they are empty the column dtype is sometimes an object so .dt raises an exception
    if not pr.review_requests.empty:
        activity_days.update(pr.review_requests[PullRequestReviewRequest.created_at.name].dt.date)
    if not pr.reviews.empty:
        activity_days.update(pr.reviews[PullRequestReview.created_at.name].dt.date)
    if not pr.comments.empty:
        activity_days.update(pr.comments[PullRequestComment.created_at.name].dt.date)
    if not pr.commits.empty:
        activity_days.update(pr.commits[PullRequestCommit.committed_date.name].dt.date)
    if sqlite:
        activity_days = [d.strftime("%Y-%m-%d") for d in sorted(activity_days)]
    else:
        # Postgres is "clever" enough to localize them otherwise
        activity_days = [
            datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) for d in activity_days
        ]
    return activity_days


def build_labels_filters(
    model: GitHubBase,
    labels: LabelFilter,
    filters: list,
    selected: Set[InstrumentedAttribute],
    postgres: bool,
) -> None:
    """Build SQL filter for labels."""
    if postgres:
        if labels.include:
            singles, multiples = LabelFilter.split(labels.include)
            or_items = []
            if singles:
                or_items.append(model.labels.has_any(singles))
            or_items.extend(model.labels.contains(m) for m in multiples)
            filters.append(or_(*or_items))
        if labels.exclude:
            filters.append(not_(model.labels.has_any(labels.exclude)))
    else:
        selected.add(model.labels)


def labels_are_compatible(
    include_singles: Set[str],
    include_multiples: List[Set[str]],
    exclude: Set[str],
    labels: Iterable[str],
) -> bool:
    """Check labels compatiblity."""
    labels = set(labels)
    return (
        include_singles.intersection(labels)
        or any(m.issubset(labels) for m in include_multiples)
        or (not include_singles and not include_multiples)
    ) and (not exclude or not exclude.intersection(labels))


def extract_release_match(
    repo: str,
    matched_bys: Dict[str, ReleaseMatch],
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
) -> Optional[str]:
    """Extract the release match for the given repo and settings."""
    try:
        matched_by = matched_bys[repo]
    except KeyError:
        return None
    return (
        release_settings.native[repo]
        .with_match(matched_by)
        .as_db(default_branches[drop_logical_repo(repo)])
    )


def triage_by_release_match(
    repo: str,
    release_match: str,
    release_settings: ReleaseSettings,
    default_branches: Dict[str, str],
    result: Any,
    ambiguous: Dict[str, Any],
) -> Optional[Any]:
    """Check the release match of the specified `repo` and return `None` if it is not effective \
    according to `release_settings`, or decide between `result` and `ambiguous`."""
    # DEV-1451: if we don't have this repository in the release settings, then it is deleted
    assert (
        repo in release_settings.native
    ), f"You must take care of deleted repositories separately: {repo}"
    if release_match in (ReleaseMatch.rejected.name, ReleaseMatch.force_push_drop.name):
        return result
    required_release_match = release_settings.native[repo]
    match_name, match_by = release_match.split("|", 1)
    match = ReleaseMatch[match_name]
    if required_release_match.match != ReleaseMatch.tag_or_branch:
        if match != required_release_match.match:
            return None
        dump = result
    else:
        try:
            dump = ambiguous[match_name]
        except KeyError:
            # event
            return None
    if match == ReleaseMatch.tag:
        target = required_release_match.tags
    elif match == ReleaseMatch.branch:
        target = required_release_match.branches.replace(
            default_branch_alias, default_branches.get(repo, default_branch_alias)
        )
    elif match == ReleaseMatch.event:
        target = required_release_match.events
    else:
        raise AssertionError("Precomputed DB may not contain Match.tag_or_branch")
    if target != match_by:
        return None
    return dump


def remove_ambiguous_prs(
    prs: PullRequestFactsMap, ambiguous: Dict[str, List[int]], matched_bys: Dict[str, ReleaseMatch]
) -> int:
    """
    Delete PRs from `prs` which are released by branch while the effective match is by tag.

    :return: Number of removed PRs.
    """
    missed = 0
    for repo, pr_node_ids in ambiguous.items():
        if matched_bys[repo] == ReleaseMatch.tag:
            for node_id in pr_node_ids:
                try:
                    del prs[(node_id, repo)]
                    missed += 1
                except KeyError:
                    continue
    return missed
