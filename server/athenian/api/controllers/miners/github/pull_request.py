import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import io
import struct
from typing import Dict, Generator, Generic, List, Mapping, Optional, Sequence, Set, TypeVar, Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy import select, sql
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, \
    PullRequestListItem, Stage
from athenian.api.models.metadata.github import Base, PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, PullRequestReviewComment, PullRequestReviewRequest, \
    Release
from athenian.api.request import acquire_conn_type, with_conn_pool


@dataclass(frozen=True)
class MinedPullRequest:
    """All the relevant information we are able to load from the metadata DB about a PR.

    All the DataFrame-s have a two-layered index:
    1. pull request id
    2. own id
    The artificial first index layer makes it is faster to select data belonging to a certain PR.
    """

    pr: pd.Series
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    review_requests: pd.DataFrame
    comments: pd.DataFrame
    commits: pd.DataFrame
    releases: pd.DataFrame


class PullRequestMiner:
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    with individual PR tuples."""

    CACHE_TTL = 5 * 60

    def __init__(self, prs: pd.DataFrame, reviews: pd.DataFrame, review_comments: pd.DataFrame,
                 review_requests: pd.DataFrame, comments: pd.DataFrame, commits: pd.DataFrame,
                 releases: pd.DataFrame):
        """Initialize a new instance of `PullRequestMiner`."""
        self._prs = prs
        self._reviews = reviews
        self._review_comments = review_comments
        self._review_requests = review_requests
        self._comments = comments
        self._commits = commits
        self._releases = releases

    def _serialize_for_cache(dfs: List[pd.DataFrame]) -> memoryview:
        assert len(dfs) < 256
        buf = io.BytesIO()
        buf.close = lambda: None  # disable closing the buffer in to_pickle()
        offsets = []
        for df in dfs:
            # pickle works 2-3x faster for both serialization and deserialization on our data
            df.to_pickle(buf)
            # tracking of the offsets is not really needed for pickles, but is required by feather
            # we don't use feather *yet* due to https://github.com/pandas-dev/pandas/issues/32587
            # FIXME(vmarkovtsev): ^^^
            offsets.append(buf.tell())
        buf.write(struct.pack("!" + "I" * len(offsets), *offsets))
        buf.write(struct.pack("!B", len(offsets)))
        return buf.getbuffer()

    def _deserialize_from_cache(data: bytes) -> List[pd.DataFrame]:
        data = memoryview(data)
        size = struct.unpack("!B", data[-1:])[0]
        offsets = (0,) + struct.unpack("!" + "I" * size, data[-size * 4 - 1:-1])
        dfs = []
        for beg, end in zip(offsets, offsets[1:]):
            df = pd.read_pickle(io.BytesIO(data[beg:end]))
            # The following code recovers the index if it was discarded.
            """
            if "pull_request_node_id" in df.columns:
                df.set_index(["pull_request_node_id", "node_id"], inplace=True)
            else:
                df.set_index("id", inplace=True)
            """
            dfs.append(df)
        return dfs

    @classmethod
    @with_conn_pool(lambda db, **_: db)
    @cached(
        exptime=lambda cls, **_: cls.CACHE_TTL,
        serialize=_serialize_for_cache,
        deserialize=_deserialize_from_cache,
        key=lambda time_from, time_to, repositories, developers, **_: (
            time_from.toordinal(),
            time_to.toordinal(),
            ",".join(sorted(repositories)),
            ",".join(sorted(developers)),
        ),
    )
    async def _mine(cls, time_from: date, time_to: date, repositories: Sequence[str],
                    developers: Sequence[str], db: databases.Database,
                    cache: Optional[aiomcache.Client], acquire_conn: acquire_conn_type,
                    ) -> List[pd.DataFrame]:
        filters = [
            sql.or_(sql.and_(PullRequest.updated_at >= time_from,
                             PullRequest.updated_at < time_to),
                    sql.and_(sql.or_(PullRequest.closed_at.is_(None),
                                     PullRequest.closed_at > time_from),
                             PullRequest.created_at < time_to)),
            PullRequest.repository_full_name.in_(repositories),
        ]
        if len(developers) > 0:
            filters.append(PullRequest.user_login.in_(developers))
        conn0 = await acquire_conn()
        prs = await read_sql_query(select([PullRequest]).where(sql.and_(*filters)),
                                   conn0, PullRequest, index="id")
        node_ids = prs[PullRequest.node_id.key] if len(prs) > 0 else set()
        # TODO(vmarkovtsev): carefully select the columns that must be fetched
        future_reviews = cls._read_filtered_models(
            conn0, PullRequestReview, node_ids, time_to)
        future_review_comments = cls._read_filtered_models(
            await acquire_conn(), PullRequestReviewComment, node_ids, time_to)
        future_review_requests = cls._read_filtered_models(
            await acquire_conn(), PullRequestReviewRequest, node_ids, time_to)
        future_comments = cls._read_filtered_models(
            await acquire_conn(), PullRequestComment, node_ids, time_to,
            columns=[PullRequestComment.created_at, PullRequestComment.user_id,
                     PullRequestComment.user_login])
        future_commits = cls._read_filtered_models(
            await acquire_conn(), PullRequestCommit, node_ids, time_to,
            columns=[PullRequestCommit.authored_date, PullRequestCommit.committed_date,
                     PullRequestCommit.author_login, PullRequestCommit.committer_login])
        reviews, review_comments, review_requests, comments, commits = await asyncio.gather(
            future_reviews, future_review_comments, future_review_requests, future_comments,
            future_commits)
        # delete from here
        releases = pd.DataFrame(columns=[
            "pull_request_node_id", "node_id", Release.created_at.key, Release.author.key])
        releases.set_index(["pull_request_node_id", "node_id"], inplace=True)
        # delete till here
        # TODO(vmarkovtsev): load releases in pain and sweat
        """
        releases = await read_sql_query(
            select([PullRequest.node_id, Release.id, Release.created_at, Release.author])
            .where(sql.and_(PullRequest.released_as == Release.id,
                            PullRequest.node_id.in_(node_ids),
                            Release.created_at < time_to)),
            conn,
            columns=[PullRequest.node_id.key, Release.id.key, Release.created_at.key,
                     Release.author.key],
            index=(PullRequest.node_id.key, Release.id.key),
        )
        """
        return [prs, reviews, review_comments, review_requests, comments, commits, releases]

    _serialize_for_cache = staticmethod(_serialize_for_cache)
    _deserialize_from_cache = staticmethod(_deserialize_from_cache)

    @classmethod
    async def mine(cls, time_from: date, time_to: date, repositories: Sequence[str],
                   developers: Sequence[str], db: databases.Database,
                   cache: Optional[aiomcache.Client]) -> "PullRequestMiner":
        """
        Create a new `PullRequestMiner` from the metadata DB according to the specified filters.

        :param time_from: Fetch PRs created starting from this date.
        :param time_to: Fetch PRs created ending with this date.
        :param repositories: PRs must belong to these repositories (prefix excluded).
        :param developers: PRs must be authored by these user IDs. An empty list means everybody.
        :param db: Metadata db instance.
        :param cache: memcached client to cache the collected data.
        """
        dfs = await cls._mine(time_from, time_to, repositories, developers, db, cache)
        return cls(*dfs)

    @staticmethod
    async def _read_filtered_models(conn: databases.core.Connection,
                                    model_cls: Base,
                                    node_ids: Sequence[str],
                                    time_to: date,
                                    columns: Optional[List[InstrumentedAttribute]] = None,
                                    ) -> pd.DataFrame:
        time_to = datetime.combine(time_to, datetime.min.time())
        df = await read_sql_query(select([model_cls]).where(
            sql.and_(model_cls.pull_request_node_id.in_(node_ids),
                     model_cls.created_at < time_to)),
            conn, model_cls, index=[model_cls.pull_request_node_id.key, model_cls.node_id.key])
        if columns is not None:
            df = df[[c.name for c in columns]]
        df.rename_axis(["pull_request_node_id", "node_id"], inplace=True)
        return df

    def __iter__(self) -> Generator[MinedPullRequest, None, None]:
        """Iterate over the individual pull requests."""
        node_id_key = PullRequest.node_id.key
        for _, pr in self._prs.iterrows():
            pr_node_id = pr[node_id_key]
            items = {}
            for k in MinedPullRequest.__dataclass_fields__:
                if k == "pr":
                    continue
                df = getattr(self, "_" + k)
                try:
                    items[k] = df.loc[pr_node_id]
                except KeyError:
                    items[k] = df.iloc[:0].copy()
            yield MinedPullRequest(pr, **items)


T = TypeVar("T")


class Fallback(Generic[T]):
    """
    A value with a "plan B".

    The idea is to return the backup in `Fallback.best` if the primary value is absent (None).
    We can check whether the primary value exists by `Fallback.value is None`.
    """

    def __init__(self, value: Optional[T], fallback: Union[None, T, "Fallback[T]"]):
        """Initialize a new instance of `Fallback`."""
        if value != value:  # NaN check
            value = None
        self.__value = value
        self.__fallback = fallback

    @property
    def best(self) -> Optional[T]:
        """The "best effort" value, either the primary or the backup one."""  # noqa: D401
        if self.__value is not None:
            return self.__value
        if isinstance(self.__fallback, Fallback):
            return self.__fallback.best
        return self.__fallback

    def __str__(self) -> str:
        """str()."""
        return "Fallback(%s, %s)" % (self.value, self.best)

    def __repr__(self) -> str:
        """repr()."""
        return "Fallback(%r, %r)" % (self.value, self.best)

    def __bool__(self) -> bool:
        """Return the value indicating whether there is any value, either primary or backup."""
        return self.best is not None

    @property
    def value(self) -> Optional[T]:
        """The primary value."""  # noqa: D401
        return self.__value

    @classmethod
    def max(cls, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate the maximum of several Fallback.best-s."""
        return cls.agg(max, *args)

    @classmethod
    def min(cls, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate the minimum of several Fallback.best-s."""
        return cls.agg(min, *args)

    @classmethod
    def agg(cls, func: callable, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate an aggregation of several Fallback.best-s."""
        try:
            return cls(func(arg.best for arg in args if arg.best is not None), None)
        except ValueError:
            return cls(None, None)


DT = Union[pd.Timestamp, datetime, None]


@dataclass(frozen=True)
class PullRequestTimes:
    """Various PR update timestamps."""

    @property
    def work_began(self) -> Fallback[DT]:  # PR_B   noqa: D102
        return Fallback.min(self.created, self.first_commit)

    created: Fallback[DT]                                # PR_C
    first_commit: Fallback[DT]                           # PR_CC
    last_commit_before_first_review: Fallback[DT]        # PR_CFR
    last_commit: Fallback[DT]                            # PR_LC
    merged: Fallback[DT]                                 # PR_M
    closed: Fallback[DT]                                 # PR_CL
    first_comment_on_first_review: Fallback[DT]          # PR_W
    first_review_request: Fallback[DT]                   # PR_S
    approved: Fallback[DT]                               # PR_A
    last_review: Fallback[DT]                            # PR_LR
    first_passed_checks: Fallback[DT]                    # PR_VS
    last_passed_checks: Fallback[DT]                     # PR_VF
    released: Fallback[DT]                               # PR_R


class ReviewResolution(Enum):
    """Possible review "state"-s in the metadata DB."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"


class PullRequestTimesMiner(PullRequestMiner):
    """Extract the pull request update timestamps from the metadata DB."""

    def _compile(self, pr: MinedPullRequest) -> PullRequestTimes:
        created_at = Fallback(pr.pr[PullRequest.created_at.key], None)
        merged_at = Fallback(pr.pr[PullRequest.merged_at.key], None)
        closed_at = Fallback(pr.pr[PullRequest.closed_at.key], None)
        first_commit = Fallback(pr.commits[PullRequestCommit.committed_date.key].min(), None)
        last_commit = Fallback(pr.commits[PullRequestCommit.committed_date.key].max(), None)
        first_comment = dtmin(
            pr.review_comments[PullRequestReviewComment.created_at.key].min(),
            pr.reviews[PullRequestReview.submitted_at.key].min(),
            pr.comments[pr.comments[PullRequestReviewComment.user_id.key]
                        != pr.pr[PullRequest.user_id.key]]  # noqa: W503
                [PullRequestReviewComment.created_at.key].min())
        if closed_at and first_comment is not None and first_comment > closed_at.best:
            first_comment = None
        first_comment_on_first_review = Fallback(first_comment, merged_at)
        if first_comment_on_first_review:
            last_commit_before_first_review = Fallback(
                pr.commits[pr.commits[PullRequestCommit.committed_date.key]
                           <= first_comment_on_first_review.best]  # noqa: W503
                    [PullRequestCommit.committed_date.key].max(),
                first_comment_on_first_review)
            # force pushes that were lost
            first_commit = Fallback.min(first_commit, last_commit_before_first_review)
            last_commit = Fallback.max(last_commit, first_commit)
        else:
            last_commit_before_first_review = last_commit
        first_review_request_backup = Fallback.min(
            Fallback.max(created_at, last_commit_before_first_review),
            first_comment_on_first_review)
        first_review_request = pr.review_requests[PullRequestReviewRequest.created_at.key].min()
        if first_review_request_backup and first_review_request == first_review_request and \
                first_review_request > first_review_request_backup.best:
            first_review_request = Fallback(None, first_review_request_backup)
        else:
            first_review_request = Fallback(first_review_request, first_review_request_backup)
        if closed_at:
            last_review = Fallback(
                pr.reviews[pr.reviews[PullRequestReview.submitted_at.key] <= closed_at.best][
                    PullRequestReview.submitted_at.key].max(),
                None)
        else:
            last_review = Fallback(pr.reviews[PullRequestReview.submitted_at.key].max(), None)
        if merged_at:
            reviews_before_merge = pr.reviews[
                pr.reviews[PullRequestReview.submitted_at.key] <= merged_at.best]
        else:
            reviews_before_merge = pr.reviews
        grouped_reviews = reviews_before_merge \
            .sort_values([PullRequestReview.submitted_at.key], ascending=True) \
            .groupby(PullRequestReview.user_id.key, sort=False) \
            .nth(0)  # the most recent review for each reviewer
        if (grouped_reviews[PullRequestReview.state.key]
                == ReviewResolution.CHANGES_REQUESTED.name).any():  # noqa: W503
            # merged with negative reviews
            approved_at_value = None
        else:
            approved_at_value = grouped_reviews[
                grouped_reviews[PullRequestReview.state.key] == ReviewResolution.APPROVED.name
            ][PullRequestReview.submitted_at.key].max()
        approved_at = Fallback(approved_at_value, merged_at)
        last_passed_checks = Fallback(None, None)  # FIXME(vmarkovtsev): no CI info
        released_at = pr.releases[Release.created_at.key].min()  # may be None or NaT - that's fine
        return PullRequestTimes(
            created=created_at,
            first_commit=first_commit,
            last_commit_before_first_review=last_commit_before_first_review,
            last_commit=last_commit,
            merged=merged_at,
            first_comment_on_first_review=first_comment_on_first_review,
            first_review_request=first_review_request,
            last_review=last_review,
            approved=approved_at,
            first_passed_checks=Fallback(None, None),  # FIXME(vmarkovtsev): no CI info
            last_passed_checks=last_passed_checks,
            released=Fallback(released_at, None),
            closed=closed_at,
        )

    def __iter__(self) -> Generator[PullRequestTimes, None, None]:
        """Iterate over the individual pull requests."""
        for pr in super().__iter__():
            yield self._compile(pr)


def dtmin(*args: Union[DT, float]) -> DT:
    """Find the minimum of several dates handling NaNs gracefully."""
    if all((arg != arg) for arg in args):
        return None
    return min(arg for arg in args if arg == arg)


class PullRequestListMiner(PullRequestTimesMiner):
    """Collect various PR metadata for displaying PRs on the frontend."""

    def __init__(self, *args, **kwargs):
        """Initialize a new instance of `PullRequestListMiner`."""
        super().__init__(*args, **kwargs)
        self._stages = set()
        self._participants = {}

    @property
    def stages(self) -> Set[Stage]:
        """Return the required PR stages."""
        return self._stages

    @stages.setter
    def stages(self, value: Sequence[Stage]):
        """Set the required PR stages."""
        self._stages = set(value)

    @property
    def participants(self) -> Dict[ParticipationKind, Set[str]]:
        """Return the required PR participants."""
        return self._participants

    @participants.setter
    def participants(self, value: Mapping[ParticipationKind, Sequence[str]]):
        """Set the required PR participants."""
        self._participants = {k: set(v) for k, v in value.items()}

    def _match_participants(self, yours: Mapping[ParticipationKind, Set[str]]) -> bool:
        """Check the PR particpants for compatibility with self.participants.

        :return: True whether the PR satisfies the participants filter, otherwise False.
        """
        if not self.participants:
            return True
        for k, v in self.participants.items():
            if yours.get(k, set()).intersection(v):
                return True
        return False

    def _compile(self, pr: MinedPullRequest) -> Optional[PullRequestListItem]:
        """Match the PR to the required participants and stages."""
        prefix = "github.com/"
        author = pr.pr[PullRequest.user_login.key]
        participants = {
            ParticipationKind.AUTHOR: {prefix + author} if author else set(),
            ParticipationKind.REVIEWER: {
                (prefix + u) for u in pr.reviews[PullRequestReview.user_login.key] if u},
            ParticipationKind.COMMENTER: {
                (prefix + u) for u in pr.comments[PullRequestComment.user_login.key] if u},
            ParticipationKind.COMMIT_COMMITTER: {
                (prefix + u) for u in pr.commits[PullRequestCommit.committer_login.key] if u},
            ParticipationKind.COMMIT_AUTHOR: {
                (prefix + u) for u in pr.commits[PullRequestCommit.author_login.key] if u},
        }
        merged_by = pr.pr[PullRequest.merged_by_login.key]
        if merged_by:
            participants[ParticipationKind.MERGER] = {prefix + merged_by}
        if not self._match_participants(participants):
            return None
        times = super()._compile(pr)
        if times.released or (times.closed and not times.merged):
            stage = Stage.DONE
        elif times.merged:
            # FIXME(vmarkovtsev): no releases data, we don't know if this is actually true
            # What the hell did you mean Vadim??? - Vadim from the future.
            stage = Stage.RELEASE
        elif times.approved:
            stage = Stage.MERGE
        elif times.first_review_request and times.last_review:
            stage = Stage.REVIEW
        else:
            stage = Stage.WIP
        if stage not in self.stages:
            return None
        return PullRequestListItem(
            repository=prefix + pr.pr[PullRequest.repository_full_name.key],
            number=pr.pr[PullRequest.number.key],
            title=pr.pr[PullRequest.title.key],
            size_added=pr.pr[PullRequest.additions.key],
            size_removed=pr.pr[PullRequest.deletions.key],
            files_changed=pr.pr[PullRequest.changed_files.key],
            created=pr.pr[PullRequest.created_at.key],
            updated=pr.pr[PullRequest.updated_at.key],
            closed=times.closed.best,
            comments=len(pr.comments),
            commits=len(pr.commits),
            review_requested=len(pr.review_requests) > 0,
            review_comments=len(pr.review_comments),
            merged=bool(times.merged),
            stage=stage,
            participants=participants,
        )

    def __iter__(self) -> Generator[PullRequestListItem, None, None]:
        """Iterate over the individual pull requests."""
        for pr in PullRequestMiner.__iter__(self):
            item = self._compile(pr)
            if item is not None:
                yield item
