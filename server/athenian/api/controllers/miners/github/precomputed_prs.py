import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import aiomcache
import databases
from dateutil.rrule import DAILY, rrule
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, delete, desc, insert, join, not_, or_, select, union_all, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.functions import coalesce

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner, \
    load_branch_commit_dates
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    fetch_precomputed_commit_history_dags, fetch_repository_commits
from athenian.api.controllers.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.controllers.miners.github.release_load import group_repos_by_release_match, \
    match_groups_to_sql
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.types import MinedPullRequest, PRParticipants, \
    PRParticipationKind, PullRequestFacts
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, ReleaseSettings
from athenian.api.db import add_pdb_hits, greatest
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestLabel, PullRequestReview, PullRequestReviewRequest, Release
from athenian.api.models.precomputed.models import GitHubBase, GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubOpenPullRequestFacts
from athenian.api.tracing import sentry_span


def _create_common_filters(time_from: Optional[datetime],
                           time_to: Optional[datetime],
                           repos: Optional[Collection[str]],
                           account: int,
                           ) -> List[ClauseElement]:
    assert isinstance(time_from, (datetime, type(None)))
    assert isinstance(time_to, (datetime, type(None)))
    ghprt = GitHubDonePullRequestFacts
    items = [
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        ghprt.acc_id == account,
    ]
    if time_to is not None:
        items.append(ghprt.pr_created_at < time_to)
    if time_from is not None:
        items.append(ghprt.pr_done_at >= time_from)
    if repos is not None:
        items.append(ghprt.repository_full_name.in_(repos))
    return items


def triage_by_release_match(repo: str,
                            release_match: str,
                            release_settings: ReleaseSettings,
                            default_branches: Dict[str, str],
                            result: Any,
                            ambiguous: Dict[str, Any],
                            ) -> Optional[Any]:
    """Check the release match of the specified `repo` and return `None` if it is not effective \
    or decide between `result` and `ambiguous`, depending on the settings."""
    # DEV-1451: if we don't have this repository in the release settings, then it is deleted
    assert repo in release_settings.native, \
        f"You must take care of deleted repositories separately: {repo}"
    if release_match in (ReleaseMatch.rejected.name,
                         ReleaseMatch.force_push_drop.name):
        return result
    required_release_match = release_settings.native[repo]
    if release_match == ReleaseMatch.event.name:
        if required_release_match.match == ReleaseMatch.event:
            return result
        return None
    match_name, match_by = release_match.split("|", 1)
    match = ReleaseMatch[match_name]
    if required_release_match.match != ReleaseMatch.tag_or_branch:
        if match != required_release_match.match:
            return None
        dump = result
    else:
        dump = ambiguous[match_name]
    if match == ReleaseMatch.tag:
        target = required_release_match.tags
    elif match == ReleaseMatch.branch:
        target = required_release_match.branches.replace(
            default_branch_alias, default_branches.get(repo, default_branch_alias))
    else:
        raise AssertionError("Precomputed DB may not contain Match.tag_or_branch")
    if target != match_by:
        return None
    return dump


@sentry_span
async def load_precomputed_done_candidates(time_from: datetime,
                                           time_to: datetime,
                                           repos: Collection[str],
                                           default_branches: Dict[str, str],
                                           release_settings: ReleaseSettings,
                                           account: int,
                                           pdb: databases.Database,
                                           ) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Load the set of done PR identifiers and specifically ambiguous PR node IDs.

    We find all the done PRs for a given time frame, repositories, and release match settings.

    :return: 1. Done PR node IDs. \
             2. Map from repository name to ambiguous PR node IDs which are released by \
             branch with tag_or_branch strategy and without tags on the time interval.
    """
    ghprt = GitHubDonePullRequestFacts
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match]
    filters = _create_common_filters(time_from, time_to, repos, account)
    with sentry_sdk.start_span(op="load_precomputed_done_candidates/fetch"):
        rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
    result = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    for row in rows:
        dump = triage_by_release_match(
            row[ghprt.repository_full_name.key], row[ghprt.release_match.key],
            release_settings, default_branches, result, ambiguous)
        if dump is None:
            continue
        dump[row[ghprt.pr_node_id.key]] = row
    result, ambiguous = _post_process_ambiguous_done_prs(result, ambiguous)
    return set(result), ambiguous


def _build_participants_filters(participants: PRParticipants,
                                filters: list,
                                selected: list,
                                postgres: bool) -> None:
    ghdprf = GitHubDonePullRequestFacts
    if postgres:
        developer_filters_single = []
        for col, pk in ((ghdprf.author, PRParticipationKind.AUTHOR),
                        (ghdprf.merger, PRParticipationKind.MERGER),
                        (ghdprf.releaser, PRParticipationKind.RELEASER)):
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            developer_filters_single.append(col.in_(col_parts))
        # do not send the same array several times
        for f in developer_filters_single[1:]:
            f.right = developer_filters_single[0].right
        developer_filters_multiple = []
        for col, pk in ((ghdprf.commenters, PRParticipationKind.COMMENTER),
                        (ghdprf.reviewers, PRParticipationKind.REVIEWER),
                        (ghdprf.commit_authors, PRParticipationKind.COMMIT_AUTHOR),
                        (ghdprf.commit_committers, PRParticipationKind.COMMIT_COMMITTER)):
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            developer_filters_multiple.append(col.has_any(col_parts))
        # do not send the same array several times
        for f in developer_filters_multiple[1:]:
            f.right = developer_filters_multiple[0].right
        filters.append(or_(*developer_filters_single, *developer_filters_multiple))
    else:
        selected.extend([
            ghdprf.author, ghdprf.merger, ghdprf.releaser, ghdprf.reviewers, ghdprf.commenters,
            ghdprf.commit_authors, ghdprf.commit_committers])


def _build_labels_filters(model: GitHubBase,
                          labels: LabelFilter,
                          filters: list,
                          selected: list,
                          postgres: bool) -> None:
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
        selected.append(model.labels)


def _labels_are_compatible(include_singles: Set[str],
                           include_multiples: List[Set[str]],
                           exclude: Set[str],
                           labels: Iterable[str]) -> bool:
    labels = set(labels)
    return ((include_singles.intersection(labels)
             or
             any(m.issubset(labels) for m in include_multiples)
             or
             (not include_singles and not include_multiples))
            and
            (not exclude or not exclude.intersection(labels)))


def _check_participants(row: Mapping, participants: PRParticipants) -> bool:
    ghprt = GitHubDonePullRequestFacts
    for col, pk in ((ghprt.author, PRParticipationKind.AUTHOR),
                    (ghprt.merger, PRParticipationKind.MERGER),
                    (ghprt.releaser, PRParticipationKind.RELEASER)):
        dev = row[col.key]
        if dev and dev in participants.get(pk, set()):
            return True
    for col, pk in ((ghprt.reviewers, PRParticipationKind.REVIEWER),
                    (ghprt.commenters, PRParticipationKind.COMMENTER),
                    (ghprt.commit_authors, PRParticipationKind.COMMIT_AUTHOR),
                    (ghprt.commit_committers, PRParticipationKind.COMMIT_COMMITTER)):
        devs = set(row[col.key])
        if devs.intersection(participants.get(pk, set())):
            return True
    return False


@sentry_span
async def load_precomputed_done_facts_filters(time_from: datetime,
                                              time_to: datetime,
                                              repos: Collection[str],
                                              participants: PRParticipants,
                                              labels: LabelFilter,
                                              default_branches: Dict[str, str],
                                              exclude_inactive: bool,
                                              release_settings: ReleaseSettings,
                                              account: int,
                                              pdb: databases.Database,
                                              ) -> Tuple[Dict[str, PullRequestFacts],
                                                         Dict[str, List[str]]]:
    """
    Fetch precomputed done PR facts.

    :return: 1. Map from PR node IDs to repo names and facts. \
             2. Map from repository name to ambiguous PR node IDs which are released by \
             branch with tag_or_branch strategy and without tags on the time interval.
    """
    ghdprf = GitHubDonePullRequestFacts
    assert time_from is not None
    assert time_to is not None
    result, ambiguous = await _load_precomputed_done_filters(
        [ghdprf.data, ghdprf.author, ghdprf.merger, ghdprf.releaser],
        time_from, time_to, repos, participants, labels,
        default_branches, exclude_inactive, release_settings, account, pdb)
    for node_id, row in result.items():
        result[node_id] = _done_pr_facts_from_row(row)
    return result, ambiguous


async def load_precomputed_done_facts_all(repos: Collection[str],
                                          default_branches: Dict[str, str],
                                          release_settings: ReleaseSettings,
                                          account: int,
                                          pdb: databases.Database,
                                          extra: Iterable[InstrumentedAttribute] = (),
                                          ) -> Tuple[Dict[str, PullRequestFacts],
                                                     Dict[str, Mapping[str, Any]]]:
    """
    Fetch all the precomputed done PR facts we have.

    We don't set the repository, the author, and the merger!

    :param extra: Additional columns to fetch.

    :return: 1. Map from PR node IDs to repo names and facts. \
             2. Map from PR node IDs to raw returned rows.
    """
    ghdprf = GitHubDonePullRequestFacts
    result, _ = await _load_precomputed_done_filters(
        [ghdprf.data, ghdprf.releaser, *extra],
        None, None, repos, {}, LabelFilter.empty(),
        default_branches, False, release_settings, account, pdb)
    raw = {}
    for node_id, row in result.items():
        result[node_id] = PullRequestFacts(
            data=row[ghdprf.data.key],
            releaser=row[ghdprf.releaser.key])
        raw[node_id] = row
    return result, raw


def _done_pr_facts_from_row(row: Mapping[str, Any]) -> PullRequestFacts:
    ghdprf = GitHubDonePullRequestFacts
    return PullRequestFacts(
        data=row[ghdprf.data.key],
        repository_full_name=row[ghdprf.repository_full_name.key],
        author=row[ghdprf.author.key],
        merger=row[ghdprf.merger.key],
        releaser=row[ghdprf.releaser.key])


@sentry_span
async def load_precomputed_done_timestamp_filters(time_from: datetime,
                                                  time_to: datetime,
                                                  repos: Collection[str],
                                                  participants: PRParticipants,
                                                  labels: LabelFilter,
                                                  default_branches: Dict[str, str],
                                                  exclude_inactive: bool,
                                                  release_settings: ReleaseSettings,
                                                  account: int,
                                                  pdb: databases.Database,
                                                  ) -> Tuple[Dict[str, datetime],
                                                             Dict[str, List[str]]]:
    """
    Fetch precomputed done PR "pr_done_at" timestamps.

    :return: 1. map from PR node IDs to their release timestamps. \
             2. Map from repository name to ambiguous PR node IDs which are released by \
             branch with tag_or_branch strategy and without tags on the time interval.
    """
    result, ambiguous = await _load_precomputed_done_filters(
        [GitHubDonePullRequestFacts.pr_done_at], time_from, time_to, repos, participants, labels,
        default_branches, exclude_inactive, release_settings, account, pdb)
    sqlite = pdb.url.dialect == "sqlite"
    for node_id, row in result.items():
        dt = row[GitHubDonePullRequestFacts.pr_done_at.key]
        if sqlite:
            dt = dt.replace(tzinfo=timezone.utc)
        result[node_id] = dt
    return result, ambiguous


def remove_ambiguous_prs(prs: Dict[str, Any],
                         ambiguous: Dict[str, List[str]],
                         matched_bys: Dict[str, ReleaseMatch]) -> int:
    """
    Delete PRs from `prs` which are released by branch while the effective match is by tag.

    :return: Number of removed PRs.
    """
    missed = 0
    for repo, pr_node_ids in ambiguous.items():
        if matched_bys[repo] == ReleaseMatch.tag:
            for node_id in pr_node_ids:
                try:
                    del prs[node_id]
                    missed += 1
                except KeyError:
                    continue
    return missed


@sentry_span
async def _load_precomputed_done_filters(columns: List[InstrumentedAttribute],
                                         time_from: Optional[datetime],
                                         time_to: Optional[datetime],
                                         repos: Collection[str],
                                         participants: PRParticipants,
                                         labels: LabelFilter,
                                         default_branches: Dict[str, str],
                                         exclude_inactive: bool,
                                         release_settings: ReleaseSettings,
                                         account: int,
                                         pdb: databases.Database,
                                         ) -> Tuple[Dict[str, Mapping[str, Any]],
                                                    Dict[str, List[str]]]:
    """
    Load some data belonging to released or rejected PRs from the precomputed DB.

    Query version. JIRA must be filtered separately.
    :return: 1. Map PR node ID -> repository name & specified column value. \
             2. Map from repository name to ambiguous PR node IDs which are released by \
             branch with tag_or_branch strategy and without tags on the time interval.
    """
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    ghprt = GitHubDonePullRequestFacts
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                ] + columns
    match_groups, event_repos, _ = group_repos_by_release_match(
        repos, default_branches, release_settings)
    match_groups[ReleaseMatch.rejected] = match_groups[ReleaseMatch.force_push_drop] = {"": repos}
    if event_repos:
        match_groups[ReleaseMatch.event] = {"": event_repos}
    or_items, _ = match_groups_to_sql(match_groups, ghprt)
    filters = _create_common_filters(time_from, time_to, None, account)
    if len(participants) > 0:
        _build_participants_filters(participants, filters, selected, postgres)
    if labels:
        _build_labels_filters(GitHubDonePullRequestFacts, labels, filters, selected, postgres)
    if exclude_inactive:
        date_range = _append_activity_days_filter(
            time_from, time_to, selected, filters, ghprt.activity_days, postgres)
    if pdb.url.dialect == "sqlite":
        query = select(selected).where(and_(or_(*or_items), *filters))
    else:
        query = union_all(*(select(selected).where(and_(item, *filters)) for item in or_items))
    with sentry_sdk.start_span(op="_load_precomputed_done_filters/fetch"):
        rows = await pdb.fetch_all(query)
    result = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    if labels and not postgres:
        include_singles, include_multiples = LabelFilter.split(labels.include)
        include_singles = set(include_singles)
        include_multiples = [set(m) for m in include_multiples]
    for row in rows:
        repo, rm = row[ghprt.repository_full_name.key], row[ghprt.release_match.key]
        dump = triage_by_release_match(
            repo, rm, release_settings, default_branches, result, ambiguous)
        if dump is None:
            continue
        if not postgres:
            if len(participants) > 0 and not _check_participants(row, participants):
                continue
            if labels and not _labels_are_compatible(include_singles, include_multiples,
                                                     labels.exclude, row[ghprt.labels.key]):
                continue
            if exclude_inactive:
                activity_days = {datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                 for d in row[ghprt.activity_days.key]}
                if not activity_days.intersection(date_range):
                    continue
        dump[row[ghprt.pr_node_id.key]] = row
    return _post_process_ambiguous_done_prs(result, ambiguous)


def _post_process_ambiguous_done_prs(result: Dict[str, Mapping[str, Any]],
                                     ambiguous: Dict[ReleaseMatch, Dict[str, Mapping[str, Any]]],
                                     ) -> Tuple[Dict[str, Mapping[str, Any]],
                                                Dict[str, List[str]]]:
    """Figure out what to do with uncertain `tag_or_branch` release matches."""
    result.update(ambiguous[ReleaseMatch.tag.name])
    repokey = GitHubDonePullRequestFacts.repository_full_name.key
    # We've found PRs released by tag belonging to these repos.
    # This means that we are going to load tags in load_releases().
    confirmed_tag_repos = {obj[repokey] for obj in ambiguous[ReleaseMatch.tag.name].values()}
    ambiguous_prs = defaultdict(list)
    for node_id, obj in ambiguous[ReleaseMatch.branch.name].items():
        if (repo := obj[repokey]) not in confirmed_tag_repos:
            result[node_id] = obj
            ambiguous_prs[repo].append(node_id)
    return result, ambiguous_prs


def build_days_range(time_from: datetime, time_to: datetime) -> Set[datetime]:
    """Build the daily range between the two provided times."""
    # timezones: date_from and date_to may be not exactly 00:00
    date_from_day = datetime.combine(
        time_from.date(), datetime.min.time(), tzinfo=timezone.utc)
    date_to_day = datetime.combine(
        time_to.date(), datetime.min.time(), tzinfo=timezone.utc)
    # date_to_day will be included
    return rrule(DAILY, dtstart=date_from_day, until=date_to_day)


def _append_activity_days_filter(time_from: datetime, time_to: datetime,
                                 selected: List[InstrumentedAttribute],
                                 filters: List[ClauseElement],
                                 activity_days_column: InstrumentedAttribute,
                                 postgres: bool) -> Set[datetime]:
    date_range = build_days_range(time_from, time_to)
    if postgres:
        filters.append(activity_days_column.overlap(list(date_range)))
    else:
        selected.append(activity_days_column)
        date_range = set(date_range)
    return date_range


@sentry_span
async def load_precomputed_done_facts_reponums(repos: Dict[str, Set[int]],
                                               default_branches: Dict[str, str],
                                               release_settings: ReleaseSettings,
                                               account: int,
                                               pdb: databases.Database,
                                               ) -> Tuple[Dict[str, PullRequestFacts],
                                                          Dict[str, List[str]]]:
    """
    Load PullRequestFacts belonging to released or rejected PRs from the precomputed DB.

    repo + numbers version.

    :return: 1. Map PR node ID -> repository name & specified column value. \
             2. Map from repository name to ambiguous PR node IDs which are released by \
             branch with tag_or_branch strategy and without tags on the time interval.
    """
    ghprt = GitHubDonePullRequestFacts
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                ghprt.data,
                ghprt.author,
                ghprt.merger,
                ghprt.releaser,
                ]
    format_version_filter = \
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg
    if pdb.url.dialect == "sqlite":
        filters = [
            format_version_filter,
            or_(*[and_(ghprt.repository_full_name == repo,
                       ghprt.number.in_(numbers),
                       ghprt.acc_id == account)
                  for repo, numbers in repos.items()]),
        ]
        query = select(selected).where(and_(*filters))
    else:
        match_groups, event_repos, _ = group_repos_by_release_match(
            repos, default_branches, release_settings)
        match_groups[ReleaseMatch.rejected] = match_groups[ReleaseMatch.force_push_drop] = \
            {"": repos}
        if event_repos:
            match_groups[ReleaseMatch.event] = {"": event_repos}
        or_items, or_repos = match_groups_to_sql(match_groups, ghprt)
        query = union_all(*(
            select(selected).where(and_(item, format_version_filter, or_(
                *[and_(ghprt.repository_full_name == repo,
                       ghprt.number.in_(repos[repo]),
                       ghprt.acc_id == account)
                  for repo in item_repos],
            )))
            for item, item_repos in zip(or_items, or_repos)))

    with sentry_sdk.start_span(op="load_precomputed_done_facts_reponums/fetch"):
        rows = await pdb.fetch_all(query)
    result = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    for row in rows:
        dump = triage_by_release_match(
            row[ghprt.repository_full_name.key], row[ghprt.release_match.key],
            release_settings, default_branches, result, ambiguous)
        if dump is None:
            continue
        dump[row[ghprt.pr_node_id.key]] = _done_pr_facts_from_row(row)
    return _post_process_ambiguous_done_prs(result, ambiguous)


@sentry_span
async def load_precomputed_done_facts_ids(node_ids: Iterable[str],
                                          default_branches: Dict[str, str],
                                          release_settings: ReleaseSettings,
                                          account: int,
                                          pdb: databases.Database,
                                          panic_on_missing_repositories: bool = True,
                                          ) -> Tuple[Dict[str, PullRequestFacts],
                                                     Dict[str, List[str]]]:
    """
    Load PullRequestFacts belonging to released or rejected PRs from the precomputed DB.

    node ID version.

    :param panic_on_missing_repositories: Whether to assert that `release_settings` contain \
      all the loaded PR repositories. If `False`, we log warnings and discard the offending PRs.

    :return: 1. Map PR node ID -> repository name & specified column value. \
             2. Map from repository name to ambiguous PR node IDs which are released by \
             branch with tag_or_branch strategy and without tags on the time interval.
    """
    log = logging.getLogger(f"{metadata.__package__}.load_precomputed_done_facts_ids")
    ghprt = GitHubDonePullRequestFacts
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                ghprt.data,
                ghprt.author,
                ghprt.merger,
                ghprt.releaser,
                ]
    filters = [
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        ghprt.pr_node_id.in_(node_ids),
        ghprt.acc_id == account,
    ]
    query = select(selected).where(and_(*filters))
    with sentry_sdk.start_span(op="load_precomputed_done_facts_ids/fetch"):
        rows = await pdb.fetch_all(query)
    result = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    for row in rows:
        repo = row[ghprt.repository_full_name.key]
        if not panic_on_missing_repositories and repo not in release_settings.native:
            log.warning("Discarding PR %s because repository %s is missing",
                        row[ghprt.pr_node_id.key], repo)
            continue
        dump = triage_by_release_match(
            repo, row[ghprt.release_match.key],
            release_settings, default_branches, result, ambiguous)
        if dump is None:
            continue
        dump[row[ghprt.pr_node_id.key]] = _done_pr_facts_from_row(row)
    return _post_process_ambiguous_done_prs(result, ambiguous)


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prs, default_branches, release_settings, **_: (
        ",".join(sorted(prs)), sorted(default_branches.items()), release_settings,
    ),
    refresh_on_access=True,
)
async def load_precomputed_pr_releases(prs: Iterable[str],
                                       time_to: datetime,
                                       matched_bys: Dict[str, ReleaseMatch],
                                       default_branches: Dict[str, str],
                                       release_settings: ReleaseSettings,
                                       account: int,
                                       pdb: databases.Database,
                                       cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    """
    Load the releases mentioned in the specified PRs.

    Each PR is represented by a node_id, a repository name, and a required release match.
    """
    log = logging.getLogger("%s.load_precomputed_pr_releases" % metadata.__package__)
    assert isinstance(time_to, datetime)
    assert time_to.tzinfo is not None
    ghprt = GitHubDonePullRequestFacts
    with sentry_sdk.start_span(op="load_precomputed_pr_releases/fetch"):
        prs = await pdb.fetch_all(
            select([ghprt.pr_node_id, ghprt.pr_done_at, ghprt.releaser, ghprt.release_url,
                    ghprt.release_node_id, ghprt.repository_full_name, ghprt.release_match])
            .where(and_(ghprt.pr_node_id.in_(prs),
                        ghprt.acc_id == account,
                        ghprt.releaser.isnot(None),
                        ghprt.pr_done_at < time_to)))
    records = []
    utc = timezone.utc
    force_push_dropped = set()
    for pr in prs:
        repo = pr[ghprt.repository_full_name.key]
        node_id = pr[ghprt.pr_node_id.key]
        release_match = pr[ghprt.release_match.key]
        if release_match in (ReleaseMatch.force_push_drop.name, ReleaseMatch.event.name):
            if release_match == ReleaseMatch.force_push_drop.name:
                if node_id in force_push_dropped:
                    continue
                force_push_dropped.add(node_id)
            records.append((node_id,
                            pr[ghprt.pr_done_at.key].replace(tzinfo=utc),
                            pr[ghprt.releaser.key].rstrip(),
                            pr[ghprt.release_url.key],
                            pr[ghprt.release_node_id.key],
                            pr[ghprt.repository_full_name.key],
                            ReleaseMatch[release_match]))
            continue
        match_name, match_by = release_match.split("|", 1)
        release_match = ReleaseMatch[match_name]
        try:
            if release_match != matched_bys[repo]:
                continue
        except KeyError:
            # pdb thinks this PR was released but our current release matching settings disagree
            log.warning("Alternative release matching detected: %s", dict(pr))
            continue
        if release_match == ReleaseMatch.tag:
            if match_by != release_settings.native[repo].tags:
                continue
        elif release_match == ReleaseMatch.branch:
            branches = release_settings.native[repo].branches.replace(
                default_branch_alias, default_branches[repo])
            if match_by != branches:
                continue
        else:
            raise AssertionError("Unsupported release match in the precomputed DB: " + match_name)
        records.append((node_id,
                        pr[ghprt.pr_done_at.key].replace(tzinfo=utc),
                        pr[ghprt.releaser.key].rstrip(),
                        pr[ghprt.release_url.key],
                        pr[ghprt.release_node_id.key],
                        pr[ghprt.repository_full_name.key],
                        release_match))
    return new_released_prs_df(records)


def _collect_activity_days(pr: MinedPullRequest, facts: PullRequestFacts, sqlite: bool):
    activity_days = set()
    if facts.released is not None:
        activity_days.add(facts.released.item().date())
    if facts.closed is not None:
        activity_days.add(facts.closed.item().date())
    activity_days.add(facts.created.item().date())
    # if they are empty the column dtype is sometimes an object so .dt raises an exception
    if not pr.review_requests.empty:
        activity_days.update(
            pr.review_requests[PullRequestReviewRequest.created_at.key].dt.date)
    if not pr.reviews.empty:
        activity_days.update(pr.reviews[PullRequestReview.created_at.key].dt.date)
    if not pr.comments.empty:
        activity_days.update(pr.comments[PullRequestComment.created_at.key].dt.date)
    if not pr.commits.empty:
        activity_days.update(pr.commits[PullRequestCommit.committed_date.key].dt.date)
    if sqlite:
        activity_days = [d.strftime("%Y-%m-%d") for d in sorted(activity_days)]
    else:
        # Postgres is "clever" enough to localize them otherwise
        activity_days = [datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
                         for d in activity_days]
    return activity_days


@sentry_span
async def store_precomputed_done_facts(prs: Iterable[MinedPullRequest],
                                       pr_facts: Iterable[Optional[PullRequestFacts]],
                                       default_branches: Dict[str, str],
                                       release_settings: ReleaseSettings,
                                       account: int,
                                       pdb: databases.Database,
                                       ) -> None:
    """Store PullRequestFacts belonging to released or rejected PRs to the precomputed DB."""
    log = logging.getLogger("%s.store_precomputed_done_facts" % metadata.__package__)
    inserted = []
    sqlite = pdb.url.dialect == "sqlite"
    for pr, facts in zip(prs, pr_facts):
        if facts is None:
            # ImpossiblePullRequest
            continue
        pr_created = pr.pr[PullRequest.created_at.key]
        try:
            assert pr_created == facts.created
        except TypeError:
            assert pr_created.to_numpy() == facts.created
        if not facts.released:
            if not (facts.force_push_dropped or (facts.closed and not facts.merged)):
                continue
            done_at = facts.closed.item().replace(tzinfo=timezone.utc)
        else:
            done_at = facts.released.item().replace(tzinfo=timezone.utc)
            if not facts.closed:
                log.error("[DEV-508] PR %s (%s#%d) is released but not closed:\n%s",
                          pr.pr[PullRequest.node_id.key],
                          pr.pr[PullRequest.repository_full_name.key],
                          pr.pr[PullRequest.number.key],
                          facts)
                continue
        repo = pr.pr[PullRequest.repository_full_name.key]
        if pr.release[matched_by_column] is not None:
            release_match = release_settings.native[repo]
            match = ReleaseMatch(pr.release[matched_by_column])
            if match == ReleaseMatch.branch:
                branch = release_match.branches.replace(
                    default_branch_alias, default_branches[repo])
                release_match = "|".join((match.name, branch))
            elif match == ReleaseMatch.tag:
                release_match = "|".join((match.name, release_match.tags))
            elif match == ReleaseMatch.force_push_drop:
                release_match = ReleaseMatch.force_push_drop.name
            elif match == ReleaseMatch.event:
                release_match = ReleaseMatch.event.name
            else:
                raise AssertionError("Unhandled release match strategy: " + match.name)
        else:
            release_match = ReleaseMatch.rejected.name
        participants = pr.participants()
        inserted.append(GitHubDonePullRequestFacts(
            acc_id=account,
            pr_node_id=pr.pr[PullRequest.node_id.key],
            release_match=release_match,
            repository_full_name=repo,
            pr_created_at=facts.created.item().replace(tzinfo=timezone.utc),
            pr_done_at=done_at,
            number=pr.pr[PullRequest.number.key],
            release_url=pr.release[Release.url.key],
            release_node_id=pr.release[Release.id.key],
            author=_flatten_set(participants[PRParticipationKind.AUTHOR]),
            merger=_flatten_set(participants[PRParticipationKind.MERGER]),
            releaser=_flatten_set(participants[PRParticipationKind.RELEASER]),
            commenters={k: "" for k in participants[PRParticipationKind.COMMENTER]},
            reviewers={k: "" for k in participants[PRParticipationKind.REVIEWER]},
            commit_authors={k: "" for k in participants[PRParticipationKind.COMMIT_AUTHOR]},
            commit_committers={k: "" for k in participants[PRParticipationKind.COMMIT_COMMITTER]},
            labels={label: "" for label in pr.labels[PullRequestLabel.name.key].values},
            activity_days=_collect_activity_days(pr, facts, sqlite),
            data=facts.data,
        ).create_defaults().explode(with_primary_keys=True))
    if not inserted:
        return
    if pdb.url.dialect in ("postgres", "postgresql"):
        sql = postgres_insert(GitHubDonePullRequestFacts)
        sql = sql.on_conflict_do_update(
            constraint=GitHubDonePullRequestFacts.__table__.primary_key,
            set_={
                GitHubDonePullRequestFacts.pr_done_at.key: sql.excluded.pr_done_at,
                GitHubDonePullRequestFacts.updated_at.key: sql.excluded.updated_at,
                GitHubDonePullRequestFacts.release_url.key: sql.excluded.release_url,
                GitHubDonePullRequestFacts.release_node_id.key: sql.excluded.release_node_id,
                GitHubDonePullRequestFacts.merger.key: sql.excluded.merger,
                GitHubDonePullRequestFacts.releaser.key: sql.excluded.releaser,
                GitHubDonePullRequestFacts.activity_days.key: sql.excluded.activity_days,
                GitHubDonePullRequestFacts.data.key: sql.excluded.data,
            },
        )
    elif pdb.url.dialect == "sqlite":
        sql = insert(GitHubDonePullRequestFacts).prefix_with("OR REPLACE")
    else:
        raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
    with sentry_sdk.start_span(op="store_precomputed_done_facts/execute_many"):
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, inserted)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, inserted)


def _flatten_set(s: set) -> Optional[Any]:
    if not s:
        return None
    assert len(s) == 1
    return next(iter(s))


@sentry_span
async def load_merged_unreleased_pull_request_facts(
        prs: pd.DataFrame,
        time_to: datetime,
        labels: LabelFilter,
        matched_bys: Dict[str, ReleaseMatch],
        default_branches: Dict[str, str],
        release_settings: ReleaseSettings,
        account: int,
        pdb: databases.Database,
        time_from: Optional[datetime] = None,
        exclude_inactive: bool = False,
) -> Dict[str, PullRequestFacts]:
    """
    Load the mapping from PR node identifiers which we are sure are not released in one of \
    `releases` to the serialized facts.

    For each merged PR we maintain the set of releases that do include that PR.

    :return: Map from PR node IDs to their facts.
    """
    if time_to != time_to:
        return {}
    assert time_to.tzinfo is not None
    if exclude_inactive:
        assert time_from is not None
    log = logging.getLogger("%s.load_merged_unreleased_pull_request_facts" % metadata.__package__)
    ghmprf = GitHubMergedPullRequestFacts
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    selected = [ghmprf.pr_node_id,
                ghmprf.repository_full_name,
                ghmprf.data,
                ghmprf.author,
                ghmprf.merger,
                ]
    default_version = ghmprf.__table__.columns[ghmprf.format_version.key].default.arg
    common_filters = [
        ghmprf.checked_until >= time_to,
        ghmprf.format_version == default_version,
        ghmprf.acc_id == account,
    ]
    if labels:
        _build_labels_filters(ghmprf, labels, common_filters, selected, postgres)
    if exclude_inactive:
        date_range = _append_activity_days_filter(
            time_from, time_to, selected, common_filters, ghmprf.activity_days, postgres)
    repos_by_match = defaultdict(list)
    for repo in prs[PullRequest.repository_full_name.key].unique():
        if (release_match := _extract_release_match(
                repo, matched_bys, default_branches, release_settings)) is None:
            # no new releases
            continue
        repos_by_match[release_match].append(repo)
    queries = []
    pr_repos = prs[PullRequest.repository_full_name.key].values.astype("S")
    pr_ids = prs.index.values
    for release_match, repos in repos_by_match.items():
        filters = [
            ghmprf.pr_node_id.in_(pr_ids[np.in1d(pr_repos, np.array(repos, dtype="S"))]),
            ghmprf.repository_full_name.in_(repos),
            ghmprf.release_match == release_match,
            *common_filters,
        ]
        queries.append(select(selected).where(and_(*filters)))
    if not queries:
        return {}
    query = union_all(*queries)
    with sentry_sdk.start_span(op="load_merged_unreleased_pr_facts/fetch"):
        rows = await pdb.fetch_all(query)
    if labels:
        include_singles, include_multiples = LabelFilter.split(labels.include)
        include_singles = set(include_singles)
        include_multiples = [set(m) for m in include_multiples]
    facts = {}
    for row in rows:
        if exclude_inactive and not postgres:
            activity_days = {datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                             for d in row[ghmprf.activity_days.key]}
            if not activity_days.intersection(date_range):
                continue
        node_id = row[ghmprf.pr_node_id.key]
        data = row[ghmprf.data.key]
        if data is None:
            # There are two known cases:
            # 1. When we load all PRs without a blacklist (/filter/pull_requests) so some merged PR
            #    is matched to releases but exists in `github_done_pull_request_facts`.
            # 2. "Impossible" PRs that are merged.
            log.warning("No precomputed facts for merged %s", node_id)
            continue
        if labels and not _labels_are_compatible(
                include_singles, include_multiples, labels.exclude, row[ghmprf.labels.key]):
            continue
        facts[node_id] = PullRequestFacts(
            data=data,
            repository_full_name=row[ghmprf.repository_full_name.key],
            author=row[ghmprf.author.key],
            merger=row[ghmprf.merger.key])
    return facts


@sentry_span
async def load_merged_pull_request_facts_all(repos: Collection[str],
                                             pr_node_id_blacklist: Collection[str],
                                             account: int,
                                             pdb: databases.Database,
                                             ) -> Dict[str, PullRequestFacts]:
    """
    Load the precomputed merged PR facts through all the time.

    We do not load the repository, the author, and the merger!

    :return: Map from PR node IDs to their facts.
    """
    log = logging.getLogger("%s.load_merged_pull_request_facts_all" % metadata.__package__)
    ghmprf = GitHubMergedPullRequestFacts
    selected = [
        ghmprf.pr_node_id,
        ghmprf.data,
    ]
    default_version = ghmprf.__table__.columns[ghmprf.format_version.key].default.arg
    filters = [
        ghmprf.pr_node_id.notin_(pr_node_id_blacklist),
        ghmprf.repository_full_name.in_(repos),
        ghmprf.format_version == default_version,
        ghmprf.acc_id == account,
    ]
    query = select(selected).where(and_(*filters))
    with sentry_sdk.start_span(op="load_merged_pull_request_facts_all/fetch"):
        rows = await pdb.fetch_all(query)
    facts = {}
    for row in rows:
        if (node_id := row[ghmprf.pr_node_id.key]) in facts:
            # different release match settings, we don't care because the facts are the same
            continue
        data = row[ghmprf.data.key]
        if data is None:
            # There are two known cases:
            # 1. When we load all PRs without a blacklist (/filter/pull_requests) so some merged PR
            #    is matched to releases but exists in `github_done_pull_request_facts`.
            # 2. "Impossible" PRs that are merged.
            log.warning("No precomputed facts for merged %s", node_id)
            continue
        facts[node_id] = PullRequestFacts(data)
    return facts


def _extract_release_match(repo: str,
                           matched_bys: Dict[str, ReleaseMatch],
                           default_branches: Dict[str, str],
                           release_settings: ReleaseSettings,
                           ) -> Optional[str]:
    try:
        matched_by = matched_bys[repo]
    except KeyError:
        return None
    release_setting = release_settings.native[repo]
    if matched_by == ReleaseMatch.tag:
        return "tag|" + release_setting.tags
    if matched_by == ReleaseMatch.branch:
        branch = release_setting.branches.replace(
            default_branch_alias, default_branches[repo])
        return "branch|" + branch
    if matched_by == ReleaseMatch.event:
        return ReleaseMatch.event.name
    raise AssertionError("Unsupported release match %s" % matched_by)


@sentry_span
async def update_unreleased_prs(merged_prs: pd.DataFrame,
                                released_prs: pd.DataFrame,
                                time_to: datetime,
                                labels: Dict[str, List[str]],
                                matched_bys: Dict[str, ReleaseMatch],
                                default_branches: Dict[str, str],
                                release_settings: ReleaseSettings,
                                account: int,
                                pdb: databases.Database,
                                unreleased_prs_event: asyncio.Event) -> None:
    """
    Bump the last check timestamps for unreleased merged PRs.

    :param merged_prs: Merged PRs to update in the pdb.
    :param released_prs: Released PRs among `merged_prs`. They should be marked as checked until \
                         the release publish time.
    :param time_to: Until when we checked releases for the specified PRs.
    """
    assert time_to.tzinfo is not None
    time_to = min(time_to, datetime.now(timezone.utc))
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    values = []
    if not released_prs.empty:
        release_times = dict(zip(released_prs.index.values,
                                 released_prs[Release.published_at.key] - timedelta(minutes=1)))
    else:
        release_times = {}
    with sentry_sdk.start_span(op="update_unreleased_prs/generate"):
        for repo, repo_prs in merged_prs.groupby(PullRequest.repository_full_name.key, sort=False):
            if (release_match := _extract_release_match(
                    repo, matched_bys, default_branches, release_settings)) is None:
                # no new releases
                continue
            for node_id, merged_at, author, merger in zip(
                    repo_prs.index.values, repo_prs[PullRequest.merged_at.key],
                    repo_prs[PullRequest.user_login.key].values,
                    repo_prs[PullRequest.merged_by_login.key].values):
                try:
                    released_time = release_times[node_id]
                except KeyError:
                    checked_until = time_to
                else:
                    if released_time == released_time:
                        checked_until = min(time_to, released_time - timedelta(seconds=1))
                    else:
                        checked_until = merged_at  # force_push_drop
                values.append(GitHubMergedPullRequestFacts(
                    acc_id=account,
                    pr_node_id=node_id,
                    release_match=release_match,
                    repository_full_name=repo,
                    checked_until=checked_until,
                    merged_at=merged_at,
                    author=author,
                    merger=merger,
                    activity_days={},
                    labels={label: "" for label in labels.get(node_id, [])},
                ).create_defaults().explode(with_primary_keys=True))
        if not values:
            unreleased_prs_event.set()
            return
        if postgres:
            sql = postgres_insert(GitHubMergedPullRequestFacts)
            sql = sql.on_conflict_do_update(
                constraint=GitHubMergedPullRequestFacts.__table__.primary_key,
                set_={
                    GitHubMergedPullRequestFacts.checked_until.key: greatest(
                        GitHubMergedPullRequestFacts.checked_until, sql.excluded.checked_until),
                    GitHubMergedPullRequestFacts.labels.key:
                        GitHubMergedPullRequestFacts.labels + sql.excluded.labels,
                    GitHubMergedPullRequestFacts.updated_at.key: sql.excluded.updated_at,
                    GitHubMergedPullRequestFacts.data.key: GitHubMergedPullRequestFacts.data,
                },
            )
        else:
            # this is wrong but we just cannot update SQLite properly
            # nothing will break though
            sql = insert(GitHubMergedPullRequestFacts).prefix_with("OR REPLACE")
    try:
        with sentry_sdk.start_span(op="update_unreleased_prs/execute"):
            if pdb.url.dialect == "sqlite":
                async with pdb.connection() as pdb_conn:
                    async with pdb_conn.transaction():
                        await pdb_conn.execute_many(sql, values)
            else:
                # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
                await pdb.execute_many(sql, values)
    finally:
        unreleased_prs_event.set()


@sentry_span
async def store_merged_unreleased_pull_request_facts(
        merged_prs_and_facts: Iterable[Tuple[MinedPullRequest, PullRequestFacts]],
        matched_bys: Dict[str, ReleaseMatch],
        default_branches: Dict[str, str],
        release_settings: ReleaseSettings,
        account: int,
        pdb: databases.Database,
        unreleased_prs_event: asyncio.Event) -> None:
    """
    Persist the facts about merged unreleased pull requests to the database.

    Each passed PR must be merged and not released, we raise an assertion otherwise.
    """
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    if not postgres:
        assert pdb.url.dialect == "sqlite"
    values = []
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for pr, facts in merged_prs_and_facts:
        assert facts.merged and not facts.released
        repo = pr.pr[PullRequest.repository_full_name.key]
        if (release_match := _extract_release_match(
                repo, matched_bys, default_branches, release_settings)) is None:
            # no new releases
            continue
        values.append(GitHubMergedPullRequestFacts(
            acc_id=account,
            pr_node_id=pr.pr[PullRequest.node_id.key],
            release_match=release_match,
            data=facts.data,
            activity_days=_collect_activity_days(pr, facts, not postgres),
            # the following does not matter, are not updated so we set to 0xdeadbeef
            repository_full_name="",
            checked_until=dt,
            merged_at=dt,
            author="",
            merger="",
            labels={},
        ).create_defaults().explode(with_primary_keys=True))
    await unreleased_prs_event.wait()
    ghmprf = GitHubMergedPullRequestFacts
    if postgres:
        sql = postgres_insert(ghmprf)
        sql = sql.on_conflict_do_update(
            constraint=ghmprf.__table__.primary_key,
            set_={
                ghmprf.data.key: sql.excluded.data,
                ghmprf.activity_days.key: sql.excluded.activity_days,
            },
        )
        with sentry_sdk.start_span(op="store_merged_unreleased_pull_request_facts/execute"):
            if pdb.url.dialect == "sqlite":
                async with pdb.connection() as pdb_conn:
                    async with pdb_conn.transaction():
                        await pdb_conn.execute_many(sql, values)
            else:
                # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
                await pdb.execute_many(sql, values)
    else:
        tasks = [
            pdb.execute(update(ghmprf).where(and_(
                ghmprf.pr_node_id == v[ghmprf.pr_node_id.key],
                ghmprf.release_match == v[ghmprf.release_match.key],
                ghmprf.format_version == v[ghmprf.format_version.key],
            )).values({ghmprf.data: v[ghmprf.data.key],
                       ghmprf.activity_days: v[ghmprf.activity_days.key],
                       ghmprf.updated_at: datetime.now(timezone.utc)})) for v in values
        ]
        await gather(*tasks)


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, participants, labels, default_branches, release_settings, **_: (  # noqa
        time_from.timestamp(), time_to.timestamp(), ",".join(sorted(repos)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        labels, sorted(default_branches.items()), release_settings,
    ),
    refresh_on_access=True,
)
async def discover_inactive_merged_unreleased_prs(time_from: datetime,
                                                  time_to: datetime,
                                                  repos: Collection[str],
                                                  participants: PRParticipants,
                                                  labels: LabelFilter,
                                                  default_branches: Dict[str, str],
                                                  release_settings: ReleaseSettings,
                                                  account: int,
                                                  pdb: databases.Database,
                                                  cache: Optional[aiomcache.Client],
                                                  ) -> Tuple[List[str], List[str]]:
    """Discover PRs which were merged before `time_from` and still not released."""
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    ghmprf = GitHubMergedPullRequestFacts
    ghdprf = GitHubDonePullRequestFacts
    selected = [ghmprf.pr_node_id,
                ghmprf.repository_full_name,
                ghmprf.release_match]
    filters = [
        coalesce(ghdprf.pr_done_at, datetime(3000, 1, 1, tzinfo=timezone.utc)) >= time_to,
        ghmprf.repository_full_name.in_(repos),
        ghmprf.merged_at < time_from,
        ghmprf.acc_id == account,
    ]
    for role, col in ((PRParticipationKind.AUTHOR, ghmprf.author),
                      (PRParticipationKind.MERGER, ghmprf.merger)):
        people = participants.get(role)
        if people:
            filters.append(col.in_(people))
    if labels:
        _build_labels_filters(ghmprf, labels, filters, selected, postgres)
    body = join(ghmprf, ghdprf, and_(
        ghdprf.acc_id == ghmprf.acc_id,
        ghdprf.pr_node_id == ghmprf.pr_node_id,
        ghdprf.release_match == ghmprf.release_match,
        ghdprf.pr_created_at < time_from,
    ), isouter=True)
    with sentry_sdk.start_span(op="load_inactive_merged_unreleased_prs/fetch"):
        rows = await pdb.fetch_all(select(selected)
                                   .select_from(body)
                                   .where(and_(*filters))
                                   .order_by(desc(GitHubMergedPullRequestFacts.merged_at)))
    ambiguous = {ReleaseMatch.tag.name: set(), ReleaseMatch.branch.name: set()}
    node_ids = []
    repos = []
    if labels and not postgres:
        include_singles, include_multiples = LabelFilter.split(labels.include)
        include_singles = set(include_singles)
        include_multiples = [set(m) for m in include_multiples]
    for row in rows:
        dump = triage_by_release_match(
            row[1], row[2], release_settings, default_branches, "whatever", ambiguous)
        if dump is None:
            continue
        # we do not care about the exact release match
        if labels and not postgres and not _labels_are_compatible(
                include_singles, include_multiples, labels.exclude,
                row[GitHubMergedPullRequestFacts.labels.key]):
            continue
        node_ids.append(row[0])
        repos.append(row[1])
    add_pdb_hits(pdb, "inactive_merged_unreleased", len(node_ids))
    return node_ids, repos


class OpenPRFactsLoader:
    """Loader for open PRs facts."""

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts(cls,
                                           prs: pd.DataFrame,
                                           account: int,
                                           pdb: databases.Database,
                                           ) -> Dict[str, PullRequestFacts]:
        """
        Fetch precomputed facts about the open PRs from the DataFrame.

        We filter open PRs inplace so the user does not have to worry about that.
        """
        open_indexes = np.nonzero(prs[PullRequest.closed_at.key].isnull().values)[0]
        node_ids = prs.index.take(open_indexes)
        authors = dict(zip(node_ids, prs[PullRequest.user_login.key].values[open_indexes]))
        ghoprf = GitHubOpenPullRequestFacts
        default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
        selected = [
            ghoprf.pr_node_id,
            ghoprf.pr_updated_at,
            ghoprf.repository_full_name,
            ghoprf.data,
        ]
        rows = await pdb.fetch_all(
            select(selected)
            .where(and_(ghoprf.pr_node_id.in_(node_ids),
                        ghoprf.format_version == default_version,
                        ghoprf.acc_id == account)))
        if not rows:
            return {}
        found_node_ids = [r[0].rstrip() for r in rows]
        found_updated_ats = [r[1] for r in rows]
        if pdb.url.dialect == "sqlite":
            found_updated_ats = [dt.replace(tzinfo=timezone.utc) for dt in found_updated_ats]
        updated_ats = prs[PullRequest.updated_at.key].take(open_indexes)
        passed_mask = updated_ats[found_node_ids] <= found_updated_ats
        passed_node_ids = set(updated_ats.index.take(np.where(passed_mask)[0]))
        facts = {}
        for row in rows:
            node_id = row[ghoprf.pr_node_id.key]
            if node_id in passed_node_ids:
                facts[node_id] = PullRequestFacts(
                    data=row[ghoprf.data.key],
                    repository_full_name=row[ghoprf.repository_full_name.key],
                    author=authors[node_id])
        return facts

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts_unfresh(cls,
                                                   prs: Iterable[str],
                                                   time_from: datetime,
                                                   time_to: datetime,
                                                   exclude_inactive: bool,
                                                   authors: Mapping[str, str],
                                                   account: int,
                                                   pdb: databases.Database,
                                                   ) -> Dict[str, PullRequestFacts]:
        """
        Fetch precomputed facts about the open PRs from the DataFrame.

        We don't filter PRs by the last update here.

        :param authors: Map from PR node IDs to their author logins.
        :return: Map from PR node IDs to their facts.
        """
        postgres = pdb.url.dialect in ("postgres", "postgresql")
        ghoprf = GitHubOpenPullRequestFacts
        selected = [ghoprf.pr_node_id, ghoprf.repository_full_name, ghoprf.data]
        default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
        filters = [
            ghoprf.pr_node_id.in_(prs),
            ghoprf.format_version == default_version,
            ghoprf.acc_id == account,
        ]
        if exclude_inactive:
            date_range = _append_activity_days_filter(
                time_from, time_to, selected, filters, ghoprf.activity_days, postgres)
        rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
        if not rows:
            return {}
        facts = {}
        for row in rows:
            if exclude_inactive and not postgres:
                activity_days = {datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                 for d in row[ghoprf.activity_days.key]}
                if not activity_days.intersection(date_range):
                    continue
            node_id = row[ghoprf.pr_node_id.key].rstrip()
            facts[node_id] = PullRequestFacts(
                data=row[ghoprf.data.key],
                repository_full_name=row[ghoprf.repository_full_name.key],
                author=authors[node_id])
        return facts

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts_all(cls,
                                               repos: Collection[str],
                                               pr_node_id_blacklist: Collection[str],
                                               account: int,
                                               pdb: databases.Database,
                                               ) -> Dict[str, PullRequestFacts]:
        """
        Load the precomputed open PR facts through all the time.

        We do not load the repository and the author!

        :return: Map from PR node IDs to their facts.
        """
        ghoprf = GitHubOpenPullRequestFacts
        selected = [
            ghoprf.pr_node_id,
            ghoprf.data,
        ]
        default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
        filters = [
            ghoprf.pr_node_id.notin_(pr_node_id_blacklist),
            ghoprf.repository_full_name.in_(repos),
            ghoprf.format_version == default_version,
            ghoprf.acc_id == account,
        ]
        query = select(selected).where(and_(*filters))
        with sentry_sdk.start_span(op="load_open_pull_request_facts_all/fetch"):
            rows = await pdb.fetch_all(query)
        facts = {row[ghoprf.pr_node_id.key]: PullRequestFacts(row[ghoprf.data.key])
                 for row in rows}
        return facts


@sentry_span
async def store_open_pull_request_facts(
        open_prs_and_facts: Iterable[Tuple[MinedPullRequest, PullRequestFacts]],
        account: int,
        pdb: databases.Database) -> None:
    """
    Persist the facts about open pull requests to the database.

    Each passed PR must be open, we raise an assertion otherwise.
    """
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    if not postgres:
        assert pdb.url.dialect == "sqlite"
    values = []
    for pr, facts in open_prs_and_facts:
        assert not facts.closed
        updated_at = pr.pr[PullRequest.updated_at.key]
        if updated_at != updated_at:
            continue
        values.append(GitHubOpenPullRequestFacts(
            acc_id=account,
            pr_node_id=pr.pr[PullRequest.node_id.key],
            repository_full_name=pr.pr[PullRequest.repository_full_name.key],
            pr_created_at=pr.pr[PullRequest.created_at.key],
            number=pr.pr[PullRequest.number.key],
            pr_updated_at=updated_at,
            activity_days=_collect_activity_days(pr, facts, not postgres),
            data=facts.data,
        ).create_defaults().explode(with_primary_keys=True))
    if postgres:
        sql = postgres_insert(GitHubOpenPullRequestFacts)
        sql = sql.on_conflict_do_update(
            constraint=GitHubOpenPullRequestFacts.__table__.primary_key,
            set_={
                GitHubOpenPullRequestFacts.pr_updated_at.key: sql.excluded.pr_updated_at,
                GitHubOpenPullRequestFacts.updated_at.key: sql.excluded.updated_at,
                GitHubOpenPullRequestFacts.data.key: sql.excluded.data,
            },
        )
    else:
        sql = insert(GitHubOpenPullRequestFacts).prefix_with("OR REPLACE")
    with sentry_sdk.start_span(op="store_open_pull_request_facts/execute"):
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, values)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, values)


@sentry_span
async def delete_force_push_dropped_prs(repos: Iterable[str],
                                        account: int,
                                        meta_ids: Tuple[int, ...],
                                        mdb: databases.Database,
                                        pdb: databases.Database,
                                        cache: Optional[aiomcache.Client],
                                        ) -> Collection[str]:
    """
    Load all released precomputed PRs and re-check that they are still accessible from \
    the branch heads. Mark inaccessible as force push dropped.

    We don't try to resolve rebased PRs here due to the intended use case.
    """
    @sentry_span
    async def fetch_branches():
        branches, _ = await BranchMiner.extract_branches(repos, meta_ids, mdb, cache)
        await load_branch_commit_dates(branches, meta_ids, mdb)
        return branches

    ghdprf = GitHubDonePullRequestFacts
    tasks = [
        pdb.fetch_all(select([ghdprf.pr_node_id])
                      .where(and_(ghdprf.repository_full_name.in_(repos),
                                  ghdprf.acc_id == account,
                                  ghdprf.release_match.like("%|%")))),
        fetch_branches(),
        fetch_precomputed_commit_history_dags(repos, account, pdb, cache),
    ]
    rows, branches, dags = await gather(*tasks, op="fetch prs + branches + dags")
    pr_node_ids = [r[0] for r in rows]
    del rows
    tasks = [
        mdb.fetch_all(select([PullRequest.merge_commit_sha, PullRequest.node_id])
                      .where(and_(PullRequest.node_id.in_(pr_node_ids),
                                  PullRequest.acc_id.in_(meta_ids)))),
        fetch_repository_commits(
            dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, True, account, meta_ids,
            mdb, pdb, cache),
    ]
    del pr_node_ids
    pr_merges, dags = await gather(*tasks, op="fetch merges + prune dags")
    accessible_hashes = np.sort(np.concatenate([dag[0] for dag in dags.values()]))
    merge_hashes = np.sort(np.fromiter((r[0] for r in pr_merges), "S40", len(pr_merges)))
    found = searchsorted_inrange(accessible_hashes, merge_hashes)
    dead_indexes = np.nonzero(accessible_hashes[found] != merge_hashes)[0]
    dead_pr_node_ids = [None] * len(dead_indexes)
    for i, dead_index in enumerate(dead_indexes):
        dead_pr_node_ids[i] = pr_merges[dead_index][1]
    del pr_merges
    with sentry_sdk.start_span(op="delete force push dropped prs",
                               description=str(len(dead_indexes))):
        await pdb.execute(
            delete(ghdprf)
            .where(and_(ghdprf.pr_node_id.in_(dead_pr_node_ids),
                        ghdprf.release_match != ReleaseMatch.force_push_drop.name)))
    return dead_pr_node_ids


# TODO: these have to be removed, these are here just for keeping backward-compatibility
# without the need to re-write already all the places these functions are called
load_open_pull_request_facts = OpenPRFactsLoader.load_open_pull_request_facts
load_open_pull_request_facts_unfresh = OpenPRFactsLoader.load_open_pull_request_facts_unfresh
load_open_pull_request_facts_all = OpenPRFactsLoader.load_open_pull_request_facts_all
