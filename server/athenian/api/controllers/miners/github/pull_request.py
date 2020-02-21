from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import io
import struct
from typing import Dict, Generator, Generic, List, Mapping, Optional, Sequence, Set, TypeVar, \
    Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy import select, sql

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import gen_cache_key
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, \
    PullRequestListItem, Stage
from athenian.api.models.metadata.github import Base, IssueComment, PullRequest, \
    PullRequestComment, PullRequestCommit, PullRequestReview


@dataclass(frozen=True)
class MinedPullRequest:
    """All the relevant information we are able to load from the metadata DB about a PR."""

    pr: pd.Series
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    comments: pd.DataFrame
    commits: pd.DataFrame


class PullRequestMiner:
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    with individual PR tuples."""

    CACHE_TTL = 5 * 60

    def __init__(self, prs: pd.DataFrame, reviews: pd.DataFrame, review_comments: pd.DataFrame,
                 comments: pd.DataFrame, commits: pd.DataFrame):
        """Initialize a new instance of `PullRequestMiner`."""
        self._prs = prs
        self._reviews = reviews
        self._review_comments = review_comments
        self._comments = comments
        self._commits = commits

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
        cache_key = None
        if cache is not None:
            cache_key = gen_cache_key(
                "PullRequestMiner|%d|%d|%s|%s",
                time_from.toordinal(),
                time_to.toordinal(),
                ",".join(sorted(repositories)),
                ",".join(sorted(developers)),
            )
            serialized = await cache.get(cache_key)
            if serialized is not None:
                prs, reviews, review_comments, comments, commits = \
                    cls._deserialize_from_cache(serialized)
                return cls(prs, reviews, review_comments, comments, commits)
        filters = [
            sql.or_(sql.and_(PullRequest.updated_at >= time_from,
                             PullRequest.updated_at < time_to),
                    sql.and_(sql.or_(PullRequest.closed_at.is_(None),
                                     PullRequest.closed_at > time_from),
                             PullRequest.created_at < time_to)),
            PullRequest.repository_fullname.in_(repositories),
        ]
        if len(developers) > 0:
            filters.append(PullRequest.user_login.in_(developers))
        async with db.connection() as conn:
            prs = await read_sql_query(select([PullRequest]).where(sql.and_(*filters)),
                                       conn, PullRequest)
            numbers = prs[PullRequest.number.key] if len(prs) > 0 else set()
            reviews = await cls._read_filtered_models(
                conn, PullRequestReview, numbers, repositories, time_to)
            review_comments = await cls._read_filtered_models(
                conn, PullRequestComment, numbers, repositories, time_to)
            comments = await cls._read_filtered_models(
                conn, IssueComment, numbers, repositories, time_to)
            commits = await cls._read_filtered_models(
                conn, PullRequestCommit, numbers, repositories, time_to)
        if cache is not None:
            await cache.set(
                cache_key,
                cls._serialize_for_cache(prs, reviews, review_comments, comments, commits),
                exptime=cls.CACHE_TTL)
        return cls(prs, reviews, review_comments, comments, commits)

    @staticmethod
    async def _read_filtered_models(conn: databases.core.Connection,
                                    model_cls: Base,
                                    numbers: Sequence[str],
                                    repositories: Sequence[str],
                                    time_to: date,
                                    ) -> pd.DataFrame:
        time_to = datetime.combine(time_to, datetime.min.time())
        return await read_sql_query(select([model_cls]).where(
            sql.and_(model_cls.pull_request_number.in_(numbers),
                     model_cls.repository_fullname.in_(repositories),
                     model_cls.created_at < time_to)),
            conn, model_cls)

    def __iter__(self) -> Generator[MinedPullRequest, None, None]:
        """Iterate over the individual pull requests."""
        number_key = PullRequest.number.key
        repo_key = PullRequest.repository_fullname.key
        for _, pr in self._prs.iterrows():
            pr_number = pr[number_key]
            pr_repo = pr[repo_key]
            items = []
            for k, model in (("reviews", PullRequestReview),
                             ("review_comments", PullRequestComment),
                             ("comments", IssueComment),
                             ("commits", PullRequestCommit)):
                attr = getattr(self, "_" + k)
                items.append(attr[
                    (attr[model.pull_request_number.key] == pr_number) &  # noqa: W504
                    (attr[model.repository_fullname.key] == pr_repo)])
            yield MinedPullRequest(pr, *items)

    @classmethod
    def _serialize_for_cache(cls, *dfs: pd.DataFrame) -> memoryview:
        assert len(dfs) < 256
        buf = io.BytesIO()
        offsets = []
        for df in dfs:
            df.to_feather(buf)
            offsets.append(buf.tell())
        buf.write(struct.pack("!" + "I" * len(offsets), *offsets))
        buf.write(struct.pack("!B", len(offsets)))
        return buf.getbuffer()

    @classmethod
    def _deserialize_from_cache(cls, data: bytes) -> List[pd.DataFrame]:
        data = memoryview(data)
        size = struct.unpack("!B", data[-1:])[0]
        offsets = (0,) + struct.unpack("!" + "I" * size, data[-size * 4 - 1:-1])
        dfs = []
        for beg, end in zip(offsets, offsets[1:]):
            buf = io.BytesIO(data[beg:end])
            dfs.append(pd.read_feather(buf))
        return dfs


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
    finalized: Fallback[DT]                              # PR_F
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
        first_commit = Fallback(pr.commits[PullRequestCommit.commit_date.key].min(), None)
        last_commit = Fallback(pr.commits[PullRequestCommit.commit_date.key].max(), None)
        first_comment_on_first_review = Fallback(
            dtmin(pr.review_comments[PullRequestComment.created_at.key].min(),
                  pr.reviews[PullRequestReview.submitted_at.key].min()),
            merged_at)
        if first_comment_on_first_review:
            last_commit_before_first_review = Fallback(
                pr.commits[pr.commits[PullRequestCommit.commit_date.key]
                           <= first_comment_on_first_review.best]  # noqa: W503
                [PullRequestCommit.commit_date.key].max(),
                first_comment_on_first_review)
            # force pushes that were lost
            first_commit = Fallback.min(first_commit, last_commit_before_first_review)
            last_commit = Fallback.max(last_commit, first_commit)
        else:
            last_commit_before_first_review = last_commit
        first_review_request = Fallback(None, Fallback.min(
            Fallback.max(created_at, last_commit_before_first_review),
            first_comment_on_first_review))  # FIXME(vmarkovtsev): no review request info
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
            grouped_reviews = reviews_before_merge \
                .sort_values([PullRequestReview.submitted_at.key], ascending=True) \
                .groupby(PullRequestReview.user_id.key, sort=False) \
                .nth(0)  # the most recent review for each reviewer
            if (grouped_reviews[PullRequestReview.state.key]
                    == ReviewResolution.CHANGES_REQUESTED).any():  # noqa: W503
                # merged with negative reviews
                approved_at_value = None
            else:
                approved_at_value = grouped_reviews[
                    grouped_reviews[PullRequestReview.state.key] == ReviewResolution.APPROVED
                ][PullRequestReview.submitted_at.key].max()
        else:
            approved_at_value = None
        approved_at = Fallback(approved_at_value, merged_at)
        last_passed_checks = Fallback(None, None)  # FIXME(vmarkovtsev): no CI info
        finalized_at = Fallback.min(
            Fallback.max(approved_at, last_passed_checks, last_commit, created_at),
            closed_at)
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
            finalized=finalized_at,
            released=Fallback(None, None),  # FIXME(vmarkovtsev): no releases
            closed=closed_at,
        )

    def __iter__(self) -> Generator[PullRequestTimes, None, None]:
        """Iterate over the individual pull requests."""
        for pr in super().__iter__():
            yield self._compile(pr)


def dtmin(first: Union[DT, float], second: Union[DT, float]) -> DT:
    """Find the minimum of two dates handling NaNs gracefully."""
    if first != first and second != second:
        return None
    if first != first:
        return second
    if second != second:
        return first
    return min(first, second)


class PullRequestListMiner(PullRequestTimesMiner):
    """Collect various PR metadata for displaying PRs on the frontend."""

    def __init__(self, prs: pd.DataFrame, reviews: pd.DataFrame, review_comments: pd.DataFrame,
                 comments: pd.DataFrame, commits: pd.DataFrame):
        """Initialize a new instance of `PullRequestListMiner`."""
        super().__init__(prs, reviews, review_comments, comments, commits)
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
            if yours[k].intersection(v):
                return True
        return False

    def _compile(self, pr: MinedPullRequest) -> Optional[PullRequestListItem]:
        """Match the PR to the required participants and stages."""
        prefix = "github.com/"
        participants = {
            ParticipationKind.AUTHOR: {prefix + pr.pr[PullRequest.user_login.key]},
            ParticipationKind.REVIEWER: {
                (prefix + u) for u in pr.reviews[PullRequestReview.user_login.key]},
            ParticipationKind.COMMENTER: {
                (prefix + u) for u in pr.comments[IssueComment.user_login.key]},
            ParticipationKind.COMMIT_COMMITTER: {
                (prefix + u) for u in pr.commits[PullRequestCommit.commiter_login.key] if u},
            ParticipationKind.COMMIT_AUTHOR: {
                (prefix + u) for u in pr.commits[PullRequestCommit.author_login.key] if u},
        }
        merged_by = pr.pr[PullRequest.merged_by_login.key]
        if merged_by:
            participants[ParticipationKind.MERGER] = {prefix + merged_by}
        if not self._match_participants(participants):
            return None
        times = super()._compile(pr)
        if times.released:
            stage = Stage.DONE
        elif times.closed:
            # FIXME(vmarkovtsev): no releases data, we don't know if this is actually true
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
            repository=prefix + pr.pr[PullRequest.repository_fullname.key],
            number=pr.pr[PullRequest.number.key],
            title=pr.pr[PullRequest.title.key],
            size_added=pr.pr[PullRequest.additions.key],
            size_removed=pr.pr[PullRequest.deletions.key],
            files_changed=pr.pr[PullRequest.changed_files.key],
            created=pr.pr[PullRequest.created_at.key],
            updated=pr.pr[PullRequest.updated_at.key],
            comments=len(pr.comments),
            commits=len(pr.commits),
            review_requested=False,  # FIXME(vmarkovtsev): no review request info
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
