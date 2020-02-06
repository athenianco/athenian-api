from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Generator, Generic, Optional, Sequence, Tuple, TypeVar, Union

import databases
import pandas as pd
from sqlalchemy import select, sql

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.models.metadata.github import Base, PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview


class PullRequestMiner:
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    with individual PR tuples."""

    def __init__(self, prs: pd.DataFrame, reviews: pd.DataFrame, review_comments: pd.DataFrame,
                 commits: pd.DataFrame, max_time: date):
        """Initialize a new instance of `PullRequestMiner`."""
        self._prs = prs
        self._reviews = reviews
        self._review_comments = review_comments
        self._commits = commits
        self.max_time = max_time

    @classmethod
    async def mine(cls, time_from: date, time_to: date, repositories: Sequence[str],
                   developers: Sequence[str], db: databases.Database) -> "PullRequestMiner":
        """
        Create a new `PullRequestMiner` from the metadata DB according to the specified filters.

        :param time_from: Fetch PRs created starting from this date.
        :param time_to: Fetch PRs created ending with this date.
        :param repositories: PRs must belong to these repositories (prefix excluded).
        :param developers: PRs must be authored by these user IDs. An empty list means everybody.
        :param db: Metadata db instance.
        """
        filters = [
            PullRequest.created_at >= time_from,
            PullRequest.created_at < time_to,
            PullRequest.repository_fullname.in_(repositories),
        ]
        if len(developers) > 0:
            filters.append(PullRequest.user_login.in_(developers))
        async with db.connection() as conn:
            prs = await read_sql_query(select([PullRequest]).where(sql.and_(*filters)),
                                       conn, PullRequest)
            numbers = prs[PullRequest.number.key] if len(prs) > 0 else set()
            reviews = await cls._read_filtered_models(
                conn, PullRequestReview, numbers, repositories)
            review_comments = await cls._read_filtered_models(
                conn, PullRequestComment, numbers, repositories)
            commits = await cls._read_filtered_models(
                conn, PullRequestCommit, numbers, repositories)
        return cls(prs, reviews, review_comments, commits, time_to)

    @staticmethod
    async def _read_filtered_models(conn: databases.core.Connection,
                                    model_cls: Base,
                                    numbers: Sequence[str],
                                    repositories: Sequence[str]) -> pd.DataFrame:
        return await read_sql_query(select([model_cls]).where(
            sql.and_(model_cls.pull_request_number.in_(numbers),
                     model_cls.repository_fullname.in_(repositories))),
            conn, model_cls)

    def __iter__(self) -> Generator[Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame],
                                    None, None]:
        """Iterate over the information collected for individual pull requests."""
        number_key = PullRequest.number.key
        repo_key = PullRequest.repository_fullname.key
        for _, pr in self._prs.iterrows():
            pr_number = pr[number_key]
            pr_repo = pr[repo_key]
            items = []
            for k, model in (("reviews", PullRequestReview),
                             ("review_comments", PullRequestComment),
                             ("commits", PullRequestCommit)):
                attr = getattr(self, "_" + k)
                items.append(attr[
                    (attr[model.pull_request_number.key] == pr_number) &  # noqa: W504
                    (attr[model.repository_fullname.key] == pr_repo)])
            yield (pr, *items)


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


DT = Union[pd.Timestamp, date, None]


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

    def truncate(self, max_time: date) -> "PullRequestTimes[DT]":
        """Erase all timestamps that go after the specified date."""
        kwargs = {}
        for key, val in vars(self).items():
            if val.best is not None and val.best > max_time:
                val = Fallback(None, None)
            kwargs[key] = val
        return PullRequestTimes(**kwargs)


class ReviewResolution(Enum):
    """Possible review "state"-s in the metadata DB."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"


class PullRequestTimesMiner(PullRequestMiner):
    """Extract the pull request update timestamps from the metadata DB."""

    def _compile(self, pr: pd.Series, reviews: pd.DataFrame, review_comments: pd.DataFrame,
                 commits: pd.DataFrame) -> PullRequestTimes:
        created_at = Fallback(pr[PullRequest.created_at.key], None)
        merged_at = Fallback(pr[PullRequest.merged_at.key], None)
        closed_at = Fallback(pr[PullRequest.closed_at.key], None)
        first_commit = Fallback(commits[PullRequestCommit.commit_date.key].min(), None)
        last_commit = Fallback(commits[PullRequestCommit.commit_date.key].max(), None)
        first_comment_on_first_review = Fallback(
            dtmin(review_comments[PullRequestComment.created_at.key].min(),
                  reviews[PullRequestReview.submitted_at.key].min()),
            merged_at)
        if first_comment_on_first_review:
            last_commit_before_first_review = Fallback(
                commits[commits[PullRequestCommit.commit_date.key]
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
                reviews[reviews[PullRequestReview.submitted_at.key] <= closed_at.best][
                    PullRequestReview.submitted_at.key].max(),
                None)
        else:
            last_review = Fallback(reviews[PullRequestReview.submitted_at.key].max(), None)
        if merged_at:
            reviews_before_merge = reviews[
                reviews[PullRequestReview.submitted_at.key] <= merged_at.best]
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
        ).truncate(self.max_time)

    def __iter__(self) -> Generator[PullRequestTimes, None, None]:
        """Iterate over the update timestamps collected for individual pull requests."""
        for pr, reviews, review_comments, commits in super().__iter__():
            yield self._compile(pr, reviews, review_comments, commits)


def dtmin(first: Union[DT, float], second: Union[DT, float]) -> DT:
    """Find the minimum of two dates handling NaNs gracefully."""
    if first != first and second != second:
        return None
    if first != first:
        return second
    if second != second:
        return first
    return min(first, second)
