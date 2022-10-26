from datetime import timedelta, timezone
from functools import partial
import random

import factory
import faker
import numpy as np
import pandas as pd

from athenian.api.internal.miners.types import PullRequestFacts

_faker = faker.Faker()
_dt_between = partial(_faker.date_time_between, tzinfo=timezone.utc)


class PullRequestFactsFactory(factory.Factory):
    class Meta:
        model = PullRequestFacts.from_fields

    force_push_dropped = False
    release_ignored = False

    @factory.lazy_attribute
    def created(self) -> pd.Timestamp:
        return pd.Timestamp(_dt_between(start_date="-3y", end_date="-6M"))

    @factory.lazy_attribute
    def first_commit(self) -> pd.Timestamp:
        return pd.Timestamp(_dt_between(start_date="-3y1M", end_date=self.created))

    @factory.lazy_attribute
    def work_began(self) -> pd.Timestamp:
        return self.first_commit

    @factory.lazy_attribute
    def last_commit_before_first_review(self) -> pd.Timestamp:
        return pd.Timestamp(
            _dt_between(start_date=self.created, end_date=self.created + timedelta(days=30)),
        )

    @factory.lazy_attribute
    def first_comment_on_first_review(self):
        return pd.Timestamp(
            _dt_between(
                start_date=self.last_commit_before_first_review, end_date=timedelta(days=2),
            ),
        )

    @factory.lazy_attribute
    def approved(self):
        if self.first_comment_on_first_review is None:
            return None
        return pd.Timestamp(
            _dt_between(
                start_date=self.first_comment_on_first_review + timedelta(days=1),
                end_date=self.first_comment_on_first_review + timedelta(days=30),
            ),
        )

    @factory.lazy_attribute
    def last_commit(self):
        start_date = (self.first_comment_on_first_review or self.created) + timedelta(days=1)
        end_date = (self.approved or start_date) + timedelta(days=30)
        return pd.Timestamp(_dt_between(start_date=start_date, end_date=end_date))

    @factory.lazy_attribute
    def merged(self):
        if self.approved is None:
            return None
        return pd.Timestamp(_dt_between(self.approved, self.approved + timedelta(days=2)))

    @factory.lazy_attribute
    def closed(self):
        return self.merged

    @factory.lazy_attribute
    def first_review_request_exact(self):
        return _dt_between(
            start_date=self.last_commit_before_first_review,
            end_date=self.first_comment_on_first_review,
        )

    @factory.lazy_attribute
    def first_review_request(self):
        if self.first_review_request_exact is None:
            return None
        return pd.Timestamp(self.first_review_request_exact)

    @factory.lazy_attribute
    def last_review(self):
        if self.approved is None or self.closed is None:
            return None
        return pd.Timestamp(_dt_between(self.approved, self.closed))

    @factory.lazy_attribute
    def released(self):
        if self.merged is None:
            return None
        return pd.Timestamp(_dt_between(self.merged, self.merged + timedelta(days=30)))

    @factory.lazy_attribute
    def size(self):
        return random.randint(10, 1000)

    @factory.lazy_attribute
    def done(self):
        return self.released

    @factory.lazy_attribute
    def review_comments(self) -> int:
        return max(0, random.randint(-5, 15))

    @factory.lazy_attribute
    def regular_comments(self):
        return max(0, random.randint(-5, 15))

    @factory.lazy_attribute
    def participants(self):
        return max(random.randint(-1, 4), 1)

    @factory.lazy_attribute
    def reviews(self):
        if self.created is None or self.last_review is None:
            values = []
        else:
            values = [
                _faker.date_time_between(self.created, self.last_review)
                for _ in range(random.randint(0, 3))
            ]
        return np.array(values, dtype="datetime64[ns]")

    @factory.lazy_attribute
    def activity_days(self):
        return np.unique(
            np.array(
                [
                    dt.replace(tzinfo=None)
                    for dt in [
                        self.created,
                        self.closed,
                        self.released,
                        self.first_review_request,
                        self.first_commit,
                        self.last_commit_before_first_review,
                        self.last_commit,
                    ]
                    if dt is not None
                ]
                + self.reviews.tolist(),
                dtype="datetime64[D]",
            ).astype("datetime64[ns]"),
        )

    @factory.lazy_attribute
    def merged_with_failed_check_runs(self):
        return ["flake8"] if random.random() > 0.9 else []
