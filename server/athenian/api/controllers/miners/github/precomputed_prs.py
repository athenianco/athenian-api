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
from sqlalchemy import and_, desc, insert, join, not_, or_, select, union_all, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.types import MinedPullRequest, Participants, \
    ParticipationKind, PullRequestFacts
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, greatest
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestLabel, PullRequestReview, PullRequestReviewRequest, Release
from athenian.api.models.precomputed.models import Base, GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubOpenPullRequestFacts
from athenian.api.tracing import sentry_span


def _create_common_filters(time_from: datetime,
                           time_to: datetime,
                           repos: Collection[str]) -> List[ClauseElement]:
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    ghprt = GitHubDonePullRequestFacts
    return [
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        ghprt.repository_full_name.in_(repos),
        ghprt.pr_created_at < time_to,
        ghprt.pr_done_at >= time_from,
    ]


def _check_release_match(repo: str,
                         release_match: str,
                         release_settings: Dict[str, ReleaseMatchSetting],
                         default_branches: Dict[str, str],
                         prefix: str,
                         result: Any,
                         ambiguous: Dict[str, Any]) -> Optional[Any]:
    if release_match in (ReleaseMatch.rejected.name, ReleaseMatch.force_push_drop.name):
        dump = result
    else:
        match_name, match_by = release_match.split("|", 1)
        match = ReleaseMatch[match_name]
        required_release_match = release_settings[prefix + repo]
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
                                           release_settings: Dict[str, ReleaseMatchSetting],
                                           pdb: databases.Database,
                                           ) -> Set[str]:
    """
    Load the set of released PR identifiers.

    We find all the released PRs for a given time frame, repositories, and release match settings.
    """
    ghprt = GitHubDonePullRequestFacts
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match]
    filters = _create_common_filters(time_from, time_to, repos)
    with sentry_sdk.start_span(op="load_precomputed_done_candidates/fetch"):
        rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
    prefix = PREFIXES["github"]
    result = set()
    ambiguous = {ReleaseMatch.tag.name: set(), ReleaseMatch.branch.name: set()}
    for row in rows:
        dump = _check_release_match(
            row[1], row[2], release_settings, default_branches, prefix, result, ambiguous)
        if dump is None:
            continue
        dump.add(row[0])
    result.update(ambiguous[ReleaseMatch.tag.name])
    result.update(ambiguous[ReleaseMatch.branch.name])
    return result


def _build_participants_filters(participants: Participants,
                                filters: list,
                                selected: list,
                                postgres: bool) -> None:
    ghdprf = GitHubDonePullRequestFacts
    if postgres:
        developer_filters_single = []
        for col, pk in ((ghdprf.author, ParticipationKind.AUTHOR),
                        (ghdprf.merger, ParticipationKind.MERGER),
                        (ghdprf.releaser, ParticipationKind.RELEASER)):
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            developer_filters_single.append(col.in_(col_parts))
        # do not send the same array several times
        for f in developer_filters_single[1:]:
            f.right = developer_filters_single[0].right
        developer_filters_multiple = []
        for col, pk in ((ghdprf.commenters, ParticipationKind.COMMENTER),
                        (ghdprf.reviewers, ParticipationKind.REVIEWER),
                        (ghdprf.commit_authors, ParticipationKind.COMMIT_AUTHOR),
                        (ghdprf.commit_committers, ParticipationKind.COMMIT_COMMITTER)):
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


def _build_labels_filters(model: Base,
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


def _check_participants(row: Mapping, participants: Participants) -> bool:
    ghprt = GitHubDonePullRequestFacts
    for col, pk in ((ghprt.author, ParticipationKind.AUTHOR),
                    (ghprt.merger, ParticipationKind.MERGER),
                    (ghprt.releaser, ParticipationKind.RELEASER)):
        dev = row[col.key]
        if dev and dev in participants.get(pk, set()):
            return True
    for col, pk in ((ghprt.reviewers, ParticipationKind.REVIEWER),
                    (ghprt.commenters, ParticipationKind.COMMENTER),
                    (ghprt.commit_authors, ParticipationKind.COMMIT_AUTHOR),
                    (ghprt.commit_committers, ParticipationKind.COMMIT_COMMITTER)):
        devs = set(row[col.key])
        if devs.intersection(participants.get(pk, set())):
            return True
    return False


@sentry_span
async def load_precomputed_done_facts_filters(time_from: datetime,
                                              time_to: datetime,
                                              repos: Collection[str],
                                              participants: Participants,
                                              labels: LabelFilter,
                                              default_branches: Dict[str, str],
                                              exclude_inactive: bool,
                                              release_settings: Dict[str, ReleaseMatchSetting],
                                              pdb: databases.Database,
                                              ) -> Dict[str, Tuple[str, PullRequestFacts]]:
    """
    Fetch precomputed done PR facts.

    :return: map from node IDs to repo name and facts.
    """
    result = await _load_precomputed_done_filters(
        GitHubDonePullRequestFacts.data, time_from, time_to, repos, participants, labels,
        default_branches, exclude_inactive, release_settings, pdb)
    for node_id, (repo, data) in result.items():
        result[node_id] = repo, pickle.loads(data)
    return result


@sentry_span
async def load_precomputed_done_timestamp_filters(time_from: datetime,
                                                  time_to: datetime,
                                                  repos: Collection[str],
                                                  participants: Participants,
                                                  labels: LabelFilter,
                                                  default_branches: Dict[str, str],
                                                  exclude_inactive: bool,
                                                  release_settings: Dict[str, ReleaseMatchSetting],
                                                  pdb: databases.Database,
                                                  ) -> Dict[str, datetime]:
    """Fetch precomputed done PR "pr_done_at" timestamps."""
    result = await _load_precomputed_done_filters(
        GitHubDonePullRequestFacts.pr_done_at, time_from, time_to, repos, participants, labels,
        default_branches, exclude_inactive, release_settings, pdb)
    sqlite = pdb.url.dialect == "sqlite"
    for node_id, (_, dt) in result.items():
        if sqlite:
            dt = dt.replace(tzinfo=timezone.utc)
        result[node_id] = dt
    return result


@sentry_span
async def _load_precomputed_done_filters(column: InstrumentedAttribute,
                                         time_from: datetime,
                                         time_to: datetime,
                                         repos: Collection[str],
                                         participants: Participants,
                                         labels: LabelFilter,
                                         default_branches: Dict[str, str],
                                         exclude_inactive: bool,
                                         release_settings: Dict[str, ReleaseMatchSetting],
                                         pdb: databases.Database,
                                         ) -> Dict[str, Tuple[str, Any]]:
    """
    Load some data belonging to released or rejected PRs from the precomputed DB.

    Query version. JIRA must be filtered separately.
    :return: Map PR node ID -> repository name & specified column value.
    """
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    ghprt = GitHubDonePullRequestFacts
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                column]
    # FIXME(vmarkovtsev): rewrite the releases matching to cluster by value and exec on the server
    filters = _create_common_filters(time_from, time_to, repos)
    if len(participants) > 0:
        _build_participants_filters(participants, filters, selected, postgres)
    if labels:
        _build_labels_filters(GitHubDonePullRequestFacts, labels, filters, selected, postgres)
    if exclude_inactive:
        # timezones: date_from and date_to may be not exactly 00:00
        date_from_day = datetime.combine(
            time_from.date(), datetime.min.time(), tzinfo=timezone.utc)
        date_to_day = datetime.combine(
            time_to.date(), datetime.min.time(), tzinfo=timezone.utc)
        # date_to_day will be included
        date_range = rrule(DAILY, dtstart=date_from_day, until=date_to_day)
        if postgres:
            filters.append(ghprt.activity_days.overlap(list(date_range)))
        else:
            selected.append(ghprt.activity_days)
            date_range = set(date_range)
    with sentry_sdk.start_span(op="_load_precomputed_done_filters/fetch"):
        rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
    prefix = PREFIXES["github"]
    result = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    if labels and not postgres:
        include_singles, include_multiples = LabelFilter.split(labels.include)
        include_singles = set(include_singles)
        include_multiples = [set(m) for m in include_multiples]
    for row in rows:
        repo, rm = row[ghprt.repository_full_name.key], row[ghprt.release_match.key]
        dump = _check_release_match(
            repo, rm, release_settings, default_branches, prefix, result, ambiguous)
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
        dump[row[ghprt.pr_node_id.key]] = repo, row[column.key]
    result.update(ambiguous[ReleaseMatch.tag.name])
    for node_id, smth in ambiguous[ReleaseMatch.branch.name].items():
        if node_id not in result:
            result[node_id] = smth
    return result


@sentry_span
async def load_precomputed_done_facts_reponums(prs: Dict[str, Set[int]],
                                               default_branches: Dict[str, str],
                                               release_settings: Dict[str, ReleaseMatchSetting],
                                               pdb: databases.Database,
                                               ) -> Dict[str, Tuple[str, PullRequestFacts]]:
    """
    Load PullRequestFacts belonging to released or rejected PRs from the precomputed DB.

    repo + numbers version.
    """
    ghprt = GitHubDonePullRequestFacts
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                ghprt.data]
    filters = [
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        or_(*[and_(ghprt.repository_full_name == repo, ghprt.number.in_(numbers))
              for repo, numbers in prs.items()]),
    ]
    with sentry_sdk.start_span(op="load_precomputed_done_facts_reponums/fetch"):
        rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
    prefix = PREFIXES["github"]
    result = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    for row in rows:
        repo = row[ghprt.repository_full_name.key]
        dump = _check_release_match(
            repo, row[ghprt.release_match.key], release_settings, default_branches, prefix,
            result, ambiguous)
        if dump is None:
            continue
        dump[row[ghprt.pr_node_id.key]] = repo, pickle.loads(row[ghprt.data.key])
    result.update(ambiguous[ReleaseMatch.tag.name])
    for node_id, facts in ambiguous[ReleaseMatch.branch.name].items():
        if node_id not in result:
            result[node_id] = facts
    return result


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
                                       release_settings: Dict[str, ReleaseMatchSetting],
                                       pdb: databases.Database,
                                       cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    """
    Load the releases mentioned in the specified PRs.

    Each PR is represented by a node_id, a repository name, and a required release match.
    """
    log = logging.getLogger("%s.load_precomputed_pr_releases" % metadata.__package__)
    ghprt = GitHubDonePullRequestFacts
    with sentry_sdk.start_span(op="load_precomputed_pr_releases/fetch"):
        prs = await pdb.fetch_all(
            select([ghprt.pr_node_id, ghprt.pr_done_at, ghprt.releaser, ghprt.release_url,
                    ghprt.release_node_id, ghprt.repository_full_name, ghprt.release_match])
            .where(and_(ghprt.pr_node_id.in_(prs),
                        ghprt.releaser.isnot(None),
                        ghprt.pr_done_at < time_to)))
    prefix = PREFIXES["github"]
    records = []
    utc = timezone.utc
    for pr in prs:
        repo = pr[ghprt.repository_full_name.key]
        match_name, match_by = pr[ghprt.release_match.key].split("|", 1)
        release_match = ReleaseMatch[match_name]
        try:
            if release_match != matched_bys[repo]:
                continue
        except KeyError:
            # pdb thinks this PR was released but our current release matching settings disagree
            log.warning("Alternative release matching detected: %s", dict(pr))
            continue
        if release_match == ReleaseMatch.tag:
            if match_by != release_settings[prefix + repo].tags:
                continue
        elif release_match == ReleaseMatch.branch:
            branches = release_settings[prefix + repo].branches.replace(
                default_branch_alias, default_branches[repo])
            if match_by != branches:
                continue
        else:
            raise AssertionError("Unsupported release match in the precomputed DB: " + match_name)
        records.append((pr[ghprt.pr_node_id.key], pr[ghprt.pr_done_at.key].replace(tzinfo=utc),
                        pr[ghprt.releaser.key].rstrip(), pr[ghprt.release_url.key],
                        pr[ghprt.release_node_id.key], pr[ghprt.repository_full_name.key],
                        ReleaseMatch[pr[ghprt.release_match.key].split("|", 1)[0]]))
    return new_released_prs_df(records)


@sentry_span
async def store_precomputed_done_facts(prs: Iterable[MinedPullRequest],
                                       pr_facts: Iterable[Optional[Tuple[Any, PullRequestFacts]]],
                                       default_branches: Dict[str, str],
                                       release_settings: Dict[str, ReleaseMatchSetting],
                                       pdb: databases.Database,
                                       ) -> None:
    """Store PullRequestFacts belonging to released or rejected PRs to the precomputed DB."""
    log = logging.getLogger("%s.store_precomputed_done_facts" % metadata.__package__)
    inserted = []
    prefix = PREFIXES["github"]
    sqlite = pdb.url.dialect == "sqlite"
    for pr, (_, facts) in zip(prs, pr_facts):
        if facts is None:
            # ImpossiblePullRequest
            continue
        assert pr.pr[PullRequest.created_at.key] == facts.created
        activity_days = set()
        if not facts.released:
            if not (facts.force_push_dropped or (facts.closed and not facts.merged)):
                continue
            done_at = facts.closed
        else:
            done_at = facts.released
            if not facts.closed:
                log.error("[DEV-508] PR %s (%s#%d) is released but not closed:\n%s",
                          pr.pr[PullRequest.node_id.key],
                          pr.pr[PullRequest.repository_full_name.key],
                          pr.pr[PullRequest.number.key],
                          facts)
                continue
            activity_days.add(facts.released.date())
        activity_days.add(facts.created.date())
        activity_days.add(facts.closed.date())
        repo = pr.pr[PullRequest.repository_full_name.key]
        if pr.release[matched_by_column] is not None:
            release_match = release_settings[prefix + repo]
            match = ReleaseMatch(pr.release[matched_by_column])
            if match == ReleaseMatch.branch:
                branch = release_match.branches.replace(
                    default_branch_alias, default_branches[repo])
                release_match = "|".join((match.name, branch))
            elif match == ReleaseMatch.tag:
                release_match = "|".join((match.name, release_match.tags))
            elif match == ReleaseMatch.force_push_drop:
                release_match = ReleaseMatch.force_push_drop.name
            else:
                raise AssertionError("Unhandled release match strategy: " + match.name)
        else:
            release_match = ReleaseMatch.rejected.name
        participants = pr.participants()
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
        inserted.append(GitHubDonePullRequestFacts(
            pr_node_id=pr.pr[PullRequest.node_id.key],
            release_match=release_match,
            repository_full_name=repo,
            pr_created_at=facts.created,
            pr_done_at=done_at,
            number=pr.pr[PullRequest.number.key],
            release_url=pr.release[Release.url.key],
            release_node_id=pr.release[Release.id.key],
            author=_flatten_set(participants[ParticipationKind.AUTHOR]),
            merger=_flatten_set(participants[ParticipationKind.MERGER]),
            releaser=_flatten_set(participants[ParticipationKind.RELEASER]),
            commenters={k: "" for k in participants[ParticipationKind.COMMENTER]},
            reviewers={k: "" for k in participants[ParticipationKind.REVIEWER]},
            commit_authors={k: "" for k in participants[ParticipationKind.COMMIT_AUTHOR]},
            commit_committers={k: "" for k in participants[ParticipationKind.COMMIT_COMMITTER]},
            labels={label: "" for label in pr.labels[PullRequestLabel.name.key].values},
            activity_days=activity_days,
            data=pickle.dumps(facts),
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
        release_settings: Dict[str, ReleaseMatchSetting],
        pdb: databases.Database) -> Dict[str, Tuple[str, PullRequestFacts]]:
    """
    Load the mapping from PR node identifiers which we are sure are not released in one of \
    `releases` to the `pickle`-d facts.

    For each merged PR we maintain the set of releases that do include that PR.
    """
    if time_to != time_to:
        return {}
    assert time_to.tzinfo is not None
    log = logging.getLogger("%s.load_merged_unreleased_pull_request_facts" % metadata.__package__)
    ghmprf = GitHubMergedPullRequestFacts
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    selected = [ghmprf.pr_node_id, ghmprf.repository_full_name, ghmprf.data]
    default_version = ghmprf.__table__.columns[ghmprf.format_version.key].default.arg
    common_filters = [
        ghmprf.checked_until >= time_to,
        ghmprf.format_version == default_version,
    ]
    if labels:
        _build_labels_filters(ghmprf, labels, common_filters, selected, postgres)
    repos_by_match = defaultdict(list)
    for repo in prs[PullRequest.repository_full_name.key].unique():
        if (release_match := _extract_release_match(
                repo, matched_bys, default_branches, release_settings)) is None:
            # no new releases
            continue
        repos_by_match[release_match].append(repo)
    queries = []
    pr_repos = prs[PullRequest.repository_full_name.key].values.astype("U")
    pr_ids = prs.index.values
    for release_match, repos in repos_by_match.items():
        filters = [
            ghmprf.pr_node_id.in_(pr_ids[np.in1d(pr_repos, np.array(repos, dtype="U"))]),
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
        facts[node_id] = row[ghmprf.repository_full_name.key], pickle.loads(data)
    return facts


def _extract_release_match(repo: str,
                           matched_bys: Dict[str, ReleaseMatch],
                           default_branches: Dict[str, str],
                           release_settings: Dict[str, ReleaseMatchSetting]) -> Optional[str]:
    try:
        matched_by = matched_bys[repo]
    except KeyError:
        return None
    release_setting = release_settings[PREFIXES["github"] + repo]
    if matched_by == ReleaseMatch.tag:
        return "tag|" + release_setting.tags
    if matched_by == ReleaseMatch.branch:
        branch = release_setting.branches.replace(
            default_branch_alias, default_branches[repo])
        return "branch|" + branch
    raise AssertionError("Unsupported release match %s" % matched_by)


@sentry_span
async def update_unreleased_prs(merged_prs: pd.DataFrame,
                                released_prs: pd.DataFrame,
                                time_to: datetime,
                                labels: Dict[str, List[str]],
                                matched_bys: Dict[str, ReleaseMatch],
                                default_branches: Dict[str, str],
                                release_settings: Dict[str, ReleaseMatchSetting],
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
                    pr_node_id=node_id,
                    release_match=release_match,
                    repository_full_name=repo,
                    checked_until=checked_until,
                    merged_at=merged_at,
                    author=author,
                    merger=merger,
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
            await pdb.execute_many(sql, values)
    finally:
        unreleased_prs_event.set()


@sentry_span
async def store_merged_unreleased_pull_request_facts(
        merged_prs_and_facts: Iterable[Tuple[Mapping[str, Any], PullRequestFacts]],
        matched_bys: Dict[str, ReleaseMatch],
        default_branches: Dict[str, str],
        release_settings: Dict[str, ReleaseMatchSetting],
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
        repo = pr[PullRequest.repository_full_name.key]
        if (release_match := _extract_release_match(
                repo, matched_bys, default_branches, release_settings)) is None:
            # no new releases
            continue
        values.append(GitHubMergedPullRequestFacts(
            pr_node_id=pr[PullRequest.node_id.key],
            release_match=release_match,
            data=pickle.dumps(facts),
            # the following does not matter, are not updated so we set to 0xdeadbeef
            repository_full_name="",
            checked_until=dt,
            merged_at=dt,
            author="",
            merger="",
            labels={},
        ).create_defaults().explode(with_primary_keys=True))
    await unreleased_prs_event.wait()
    if postgres:
        sql = postgres_insert(GitHubMergedPullRequestFacts)
        sql = sql.on_conflict_do_update(
            constraint=GitHubMergedPullRequestFacts.__table__.primary_key,
            set_={
                GitHubMergedPullRequestFacts.data.key: sql.excluded.data,
            },
        )
        with sentry_sdk.start_span(op="store_open_pull_request_facts/execute"):
            await pdb.execute_many(sql, values)
    else:
        ghmprf = GitHubMergedPullRequestFacts
        tasks = [
            pdb.execute(update(ghmprf).where(and_(
                ghmprf.pr_node_id == v[ghmprf.pr_node_id.key],
                ghmprf.release_match == v[ghmprf.release_match.key],
                ghmprf.format_version == v[ghmprf.format_version.key],
            )).values({ghmprf.data: v[ghmprf.data.key],
                       ghmprf.updated_at: datetime.now(timezone.utc)})) for v in values
        ]
        for err in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(err, Exception):
                raise err from None


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
                                                  participants: Participants,
                                                  labels: LabelFilter,
                                                  default_branches: Dict[str, str],
                                                  release_settings: Dict[str, ReleaseMatchSetting],
                                                  pdb: databases.Database,
                                                  cache: Optional[aiomcache.Client],
                                                  ) -> Tuple[List[str], List[str]]:
    """Discover PRs which were merged before `time_from` and still not released."""
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    selected = [GitHubMergedPullRequestFacts.pr_node_id,
                GitHubMergedPullRequestFacts.repository_full_name,
                GitHubMergedPullRequestFacts.release_match]
    filters = [
        or_(GitHubDonePullRequestFacts.pr_done_at.is_(None),
            GitHubDonePullRequestFacts.pr_done_at >= time_to),
        GitHubMergedPullRequestFacts.repository_full_name.in_(repos),
        GitHubMergedPullRequestFacts.merged_at < time_from,
    ]
    for role, col in ((ParticipationKind.AUTHOR, GitHubMergedPullRequestFacts.author),
                      (ParticipationKind.MERGER, GitHubMergedPullRequestFacts.merger)):
        people = participants.get(role)
        if people:
            filters.append(col.in_(people))
    if labels:
        _build_labels_filters(GitHubMergedPullRequestFacts, labels, filters, selected, postgres)
    body = join(GitHubMergedPullRequestFacts, GitHubDonePullRequestFacts, and_(
        GitHubDonePullRequestFacts.pr_node_id == GitHubMergedPullRequestFacts.pr_node_id,
        GitHubDonePullRequestFacts.release_match == GitHubMergedPullRequestFacts.release_match,
        GitHubDonePullRequestFacts.repository_full_name.in_(repos),
        GitHubDonePullRequestFacts.pr_created_at < time_from,
    ), isouter=True)
    with sentry_sdk.start_span(op="load_inactive_merged_unreleased_prs/fetch"):
        rows = await pdb.fetch_all(select(selected)
                                   .select_from(body)
                                   .where(and_(*filters))
                                   .order_by(desc(GitHubMergedPullRequestFacts.merged_at)))
    prefix = PREFIXES["github"]
    ambiguous = {ReleaseMatch.tag.name: set(), ReleaseMatch.branch.name: set()}
    node_ids = []
    repos = []
    if labels and not postgres:
        include_singles, include_multiples = LabelFilter.split(labels.include)
        include_singles = set(include_singles)
        include_multiples = [set(m) for m in include_multiples]
    for row in rows:
        dump = _check_release_match(
            row[1], row[2], release_settings, default_branches, prefix, "whatever", ambiguous)
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


@sentry_span
async def load_open_pull_request_facts(prs: pd.DataFrame,
                                       pdb: databases.Database,
                                       ) -> Dict[str, Tuple[str, PullRequestFacts]]:
    """
    Fetch precomputed facts about the open PRs from the DataFrame.

    We filter open PRs inplace so the user does not have to worry about that.
    """
    open_indexes = np.where(prs[PullRequest.closed_at.key].isnull())[0]
    node_ids = prs.index.take(open_indexes)
    ghoprf = GitHubOpenPullRequestFacts
    default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
    rows = await pdb.fetch_all(
        select([ghoprf.pr_node_id, ghoprf.pr_updated_at, ghoprf.repository_full_name, ghoprf.data])
        .where(and_(ghoprf.pr_node_id.in_(node_ids),
                    ghoprf.format_version == default_version)))
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
            facts[node_id] = (
                row[ghoprf.repository_full_name.key], pickle.loads(row[ghoprf.data.key]),
            )
    return facts


@sentry_span
async def load_open_pull_request_facts_unfresh(prs: Iterable[str],
                                               pdb: databases.Database,
                                               ) -> Dict[str, Tuple[str, PullRequestFacts]]:
    """
    Fetch precomputed facts about the open PRs from the DataFrame.

    We don't filter PRs by the last update here.
    """
    ghoprf = GitHubOpenPullRequestFacts
    default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
    rows = await pdb.fetch_all(
        select([ghoprf.pr_node_id, ghoprf.repository_full_name, ghoprf.data])
        .where(and_(ghoprf.pr_node_id.in_(prs),
                    ghoprf.format_version == default_version)))
    if not rows:
        return {}
    facts = {row[ghoprf.pr_node_id.key].rstrip(): (
        row[ghoprf.repository_full_name.key], pickle.loads(row[ghoprf.data.key]),
    ) for row in rows}
    return facts


@sentry_span
async def store_open_pull_request_facts(
        open_prs_and_facts: Iterable[Tuple[Mapping[str, Any], PullRequestFacts]],
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
        updated_at = pr[PullRequest.updated_at.key]
        if updated_at != updated_at:
            continue
        values.append(GitHubOpenPullRequestFacts(
            pr_node_id=pr[PullRequest.node_id.key],
            repository_full_name=pr[PullRequest.repository_full_name.key],
            pr_created_at=pr[PullRequest.created_at.key],
            number=pr[PullRequest.number.key],
            pr_updated_at=updated_at,
            data=pickle.dumps(facts),
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
        await pdb.execute_many(sql, values)
