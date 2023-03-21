from datetime import timedelta, timezone
import random

import factory
import faker
import numpy as np

from athenian.api.internal.miners.types import LoadedJIRADetails, PullRequestFacts

_faker = faker.Faker()


def _dt_between(*args, **kwargs):
    return _faker.date_time_between(*args, **kwargs, tzinfo=timezone.utc).replace(tzinfo=None)


class PullRequestFactsFactory(factory.Factory):
    class Meta:
        model = PullRequestFacts.from_fields

    force_push_dropped = False
    release_ignored = False
    jira = LoadedJIRADetails.empty()

    @factory.lazy_attribute
    def created(self) -> np.datetime64:
        return np.datetime64(_dt_between(start_date="-3y", end_date="-6M"), "s")

    @factory.lazy_attribute
    def first_commit(self) -> np.datetime64:
        return np.datetime64(_dt_between(start_date="-3y1M", end_date=self.created.item()), "s")

    @factory.lazy_attribute
    def work_began(self) -> np.datetime64 | None:
        return self.first_commit

    @factory.lazy_attribute
    def last_commit_before_first_review(self) -> np.datetime64:
        return np.datetime64(
            _dt_between(
                start_date=self.created.item(), end_date=self.created.item() + timedelta(days=30),
            ),
            "s",
        )

    @factory.lazy_attribute
    def first_comment_on_first_review(self) -> np.datetime64:
        return np.datetime64(
            _dt_between(
                start_date=self.last_commit_before_first_review.item(), end_date=timedelta(days=2),
            ),
            "s",
        )

    @factory.lazy_attribute
    def approved(self) -> np.datetime64 | None:
        if self.first_comment_on_first_review is None:
            return None
        return np.datetime64(
            _dt_between(
                start_date=self.first_comment_on_first_review.item() + timedelta(days=1),
                end_date=self.first_comment_on_first_review.item() + timedelta(days=30),
            ),
            "s",
        )

    @factory.lazy_attribute
    def last_commit(self) -> np.datetime64:
        start_date = (self.first_comment_on_first_review or self.created) + np.timedelta64(1, "D")
        end_date = (self.approved or start_date) + np.timedelta64(30, "D")
        return np.datetime64(
            _dt_between(start_date=start_date.item(), end_date=end_date.item()), "s",
        )

    @factory.lazy_attribute
    def merged(self) -> np.datetime64 | None:
        if self.approved is None:
            return None
        return np.datetime64(
            _dt_between(self.approved.item(), self.approved.item() + timedelta(days=2)), "s",
        )

    @factory.lazy_attribute
    def closed(self) -> np.datetime64 | None:
        return self.merged

    @factory.lazy_attribute
    def first_review_request_exact(self) -> np.datetime64:
        return np.datetime64(
            _dt_between(
                start_date=self.last_commit_before_first_review.item(),
                end_date=self.first_comment_on_first_review.item(),
            ),
            "s",
        )

    @factory.lazy_attribute
    def first_review_request(self) -> np.datetime64 | None:
        if self.first_review_request_exact is None:
            return None
        return np.datetime64(self.first_review_request_exact.item(), "s")

    @factory.lazy_attribute
    def last_review(self) -> np.datetime64 | None:
        if self.approved is None or self.closed is None:
            return None
        return np.datetime64(_dt_between(self.approved.item(), self.closed.item()), "s")

    @factory.lazy_attribute
    def released(self) -> np.datetime64 | None:
        if self.merged is None:
            return None
        return np.datetime64(
            _dt_between(self.merged.item(), self.merged.item() + timedelta(days=30)), "s",
        )

    @factory.lazy_attribute
    def size(self) -> int:
        return random.randint(10, 1000)

    @factory.lazy_attribute
    def done(self) -> np.datetime64 | None:
        return self.released

    @factory.lazy_attribute
    def review_comments(self) -> int:
        return max(0, random.randint(-5, 15))

    @factory.lazy_attribute
    def regular_comments(self) -> int:
        return max(0, random.randint(-5, 15))

    @factory.lazy_attribute
    def participants(self) -> int:
        return max(random.randint(-1, 4), 1)

    @factory.lazy_attribute
    def reviews(self) -> np.ndarray:
        if self.created is None or self.last_review is None:
            values = []
        else:
            values = [
                _dt_between(self.created.item(), self.last_review.item())
                for _ in range(random.randint(0, 3))
            ]
        return np.array(values, dtype="datetime64[us]")

    @factory.lazy_attribute
    def activity_days(self):
        return np.unique(
            np.concatenate(
                [
                    np.array(
                        [
                            dt
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
                        ],
                        dtype="datetime64[D]",
                    ),
                    self.reviews,
                ],
            ).astype("datetime64[us]"),
        )

    @factory.lazy_attribute
    def merged_with_failed_check_runs(self):
        return ["flake8"] if random.random() > 0.9 else []
