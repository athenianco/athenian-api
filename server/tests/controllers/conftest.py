from datetime import timedelta, timezone
from random import randint
import warnings

import faker
import pytest

from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.release import _empty_dag, _fetch_commit_history_edges
from athenian.api.controllers.miners.github.release_accelerated import join_dags
from athenian.api.controllers.miners.types import Fallback, PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting


@pytest.fixture(scope="function")
def no_deprecation_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture(scope="module")
def release_match_setting_tag():
    return {
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches="", tags=".*", match=ReleaseMatch.tag),
    }


@pytest.fixture(scope="module")
def default_branches():
    return {"src-d/go-git": "master"}


_branches = None


@pytest.fixture(scope="function")
async def branches(mdb):
    global _branches
    if _branches is None:
        _branches, _ = await extract_branches(["src-d/go-git"], mdb, None)
    return _branches


@pytest.fixture
def pr_samples():
    def generate(n):
        fake = faker.Faker()

        def random_pr():
            created_at = fake.date_time_between(
                start_date="-3y", end_date="-6M", tzinfo=timezone.utc)
            first_commit = fake.date_time_between(
                start_date="-3y1M", end_date=created_at, tzinfo=timezone.utc)
            last_commit_before_first_review = fake.date_time_between(
                start_date=created_at, end_date=created_at + timedelta(days=30),
                tzinfo=timezone.utc)
            first_comment_on_first_review = fake.date_time_between(
                start_date=last_commit_before_first_review, end_date=timedelta(days=2),
                tzinfo=timezone.utc)
            first_review_request = fake.date_time_between(
                start_date=last_commit_before_first_review, end_date=first_comment_on_first_review,
                tzinfo=timezone.utc)
            first_passed_checks = fake.date_time_between(
                start_date=created_at, end_date=first_review_request, tzinfo=timezone.utc)
            approved_at = fake.date_time_between(
                start_date=first_comment_on_first_review + timedelta(days=1),
                end_date=first_comment_on_first_review + timedelta(days=30),
                tzinfo=timezone.utc)
            last_commit = fake.date_time_between(
                start_date=first_comment_on_first_review + timedelta(days=1),
                end_date=approved_at,
                tzinfo=timezone.utc)
            last_passed_checks = fake.date_time_between(
                last_commit, last_commit + timedelta(days=1), tzinfo=timezone.utc)
            merged_at = fake.date_time_between(
                approved_at, approved_at + timedelta(days=2), tzinfo=timezone.utc)
            closed_at = merged_at
            last_review = fake.date_time_between(approved_at, closed_at, tzinfo=timezone.utc)
            released_at = fake.date_time_between(
                merged_at, merged_at + timedelta(days=30), tzinfo=timezone.utc)
            return PullRequestFacts(
                created=Fallback(created_at, None),
                first_commit=Fallback(first_commit, created_at),
                last_commit_before_first_review=Fallback(last_commit_before_first_review, None),
                last_commit=Fallback(last_commit, None),
                merged=Fallback(merged_at, None),
                first_comment_on_first_review=Fallback(first_comment_on_first_review, None),
                first_review_request=Fallback(first_review_request, None),
                last_review=Fallback(last_review, None),
                approved=Fallback(approved_at, None),
                first_passed_checks=Fallback(first_passed_checks, None),
                last_passed_checks=Fallback(last_passed_checks, None),
                released=Fallback(released_at, None),
                closed=Fallback(closed_at, None),
                size=randint(10, 1000),
            )

        return [random_pr() for _ in range(n)]
    return generate


_dag = None


async def fetch_dag(mdb, heads=None):
    if heads is None:
        heads = [
            "MDY6Q29tbWl0NDQ3MzkwNDQ6MTdkYmQ4ODY2MTZmODJiZTJhNTljMGQwMmZkOTNkM2Q2OWYyMzkyYw==",
        ]
    edges = await _fetch_commit_history_edges(heads, [], mdb)
    return {"src-d/go-git": join_dags(*_empty_dag(), edges)}


@pytest.fixture(scope="function")  # we cannot declare it "module" because of mdb's scope
async def dag(mdb):
    global _dag
    if _dag is not None:
        return _dag
    _dag = await fetch_dag(mdb)
    return _dag


def pytest_configure(config):
    for mark in ("filter_repositories", "filter_contributors", "filter_pull_requests",
                 "filter_commits", "filter_releases", "filter_labels"):
        config.addinivalue_line("markers", mark)
