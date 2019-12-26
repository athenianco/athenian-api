from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generic, Optional, Sequence, TypeVar, Union

import databases
import pandas as pd
from sqlalchemy import select, sql

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, PullRequestReview


class PullRequestMiner:
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    with individual PR tuples."""

    def __init__(self, prs: pd.DataFrame, reviews: pd.DataFrame, review_comments: pd.DataFrame):
        """Initialize a new instance of `PullRequestMiner`."""
        self._prs = prs
        self._reviews = reviews
        self._review_comments = review_comments

    @classmethod
    async def mine(cls, time_from: datetime, time_to: datetime, repositories: Sequence[str],
                   db: databases.Database) -> "PullRequestMiner":
        """Create a new `PullRequestMiner` from the metadata DB according to the specified \
        filters."""
        prs = await read_sql_query(select([PullRequest]).where(sql.and_(
            PullRequest.repository_name.in_(repositories),
            PullRequest.created_at >= time_from,
            PullRequest.created_at < time_to,
        )), db)
        numbers = prs["number"]
        reviews = await read_sql_query(select([PullRequestReview]).where(
            PullRequestReview.pull_request_number.in_(numbers)), db)
        review_comments = await read_sql_query(select([PullRequestComment]).where(
            PullRequestComment.pull_request_number.in_(numbers)), db)
        return cls(prs, reviews, review_comments)

    def __iter__(self):
        """Iterate over the information collected for individual pull requests."""
        for _, pr in self._prs.iterrows():
            reviews = self._reviews[self._reviews["pull_request_number"] == pr["number"]]
            review_comments = self._review_comments[
                self._review_comments["pull_request_number"] == pr["number"]]
            yield pr, reviews, review_comments


T = TypeVar("T")


class Fallback(Generic[T]):
    """
    A value with a "plan B".

    The idea is to return the backup in `Fallback.best` if the primary value is absent (None).
    We can check whether the primary value exists by `Fallback.value is None`.
    """

    def __init__(self, value: Optional[T], fallback: Union[None, T, "Fallback[T]"]):
        """Initialize a new instance of `Fallback`."""
        if value != value:
            value = None  # NaN
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


@dataclass(frozen=True)
class PullRequestTimes:
    """Various PR update timestamps."""

    created: Fallback[datetime]                                # PR_C
    first_commit: Fallback[datetime]                           # PR_CC

    @property
    def work_begins(self) -> Fallback[datetime]:               # PR_B   noqa: D102
        return Fallback.min(self.created, self.first_commit)

    last_commit_before_first_review: Fallback[datetime]        # PR_CFR
    last_commit: Fallback[datetime]                            # PR_L
    merged: Fallback[datetime]                                 # PR_M
    closed: Fallback[datetime]                                 # PR_CL
    first_comment_on_first_review: Fallback[datetime]          # PR_W
    first_review_request: Fallback[datetime]                   # PR_S
    approved: Fallback[datetime]                               # PR_A
    first_passed_checks: Fallback[datetime]                    # PR_VS
    last_passed_checks: Fallback[datetime]                     # PR_VF
    finalized: Fallback[datetime]                              # PR_F
    released: Fallback[datetime]                               # PR_R


class ReviewResolution(Enum):
    """Possible review "state"-s in the metadata DB."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"


class PullRequestTimesMiner(PullRequestMiner):
    """Extract the pull request update timestamps from the metadata DB."""

    @classmethod
    def _compile(cls, pr: pd.Series, reviews: pd.DataFrame, review_comments: pd.DataFrame):
        created_at = Fallback(pr["created_at"], None)
        merged_at = Fallback(pr["merged_at"], None)
        first_comment_on_first_review = Fallback(review_comments["created_at"].min(), merged_at)
        last_commit_before_first_review = first_comment_on_first_review  # FIXME(vmarkovtsev): no commit timestamps # noqa
        first_review_request = Fallback(None, Fallback.min(
            Fallback.max(created_at, last_commit_before_first_review),
            first_comment_on_first_review))  # FIXME(vmarkovtsev): no review request info
        if merged_at.value is not None:
            reviews_before_merge = reviews[reviews["submitted_at"] <= merged_at]
            grouped_reviews = reviews_before_merge \
                .sort_values(["submitted_at"], ascending=True) \
                .groupby("user_login", sort=False) \
                .nth(0)  # the most recent review for each reviewer
            if (grouped_reviews["state"] == ReviewResolution.CHANGES_REQUESTED).any():
                # merged with negative reviews
                approved_at_value = None
            else:
                approved_at_value = grouped_reviews[
                    grouped_reviews["state"] == ReviewResolution.APPROVED]["submitted_at"].max()
            approved_at = Fallback(approved_at_value, None)
        else:
            approved_at = Fallback(None, None)
        last_commit = Fallback(None, created_at)  # FIXME(vmarkovtsev): no commit info
        last_passed_checks = Fallback(None, None)  # FIXME(vmarkovtsev): no CI info
        closed_at = Fallback(pr["closed_at"], None)
        return PullRequestTimes(
            created=created_at,
            first_commit=Fallback(None, created_at),  # FIXME(vmarkovtsev): no commit info
            last_commit_before_first_review=last_commit_before_first_review,
            last_commit=last_commit,
            merged=merged_at,
            first_comment_on_first_review=first_comment_on_first_review,
            first_review_request=first_review_request,
            approved=approved_at,
            first_passed_checks=Fallback(None, None),  # FIXME(vmarkovtsev): no CI info
            last_passed_checks=last_passed_checks,
            finalized=Fallback.min(Fallback.max(approved_at, last_passed_checks, last_commit),
                                   closed_at),
            released=Fallback(None, None),  # FIXME(vmarkovtsev): no releases
            closed=closed_at,
        )

    def __iter__(self):
        """Iterate over the update timestamps collected for individual pull requests."""
        for pr, reviews, review_comments in super().__iter__():
            yield self._compile(pr, reviews, review_comments)
