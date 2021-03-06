from datetime import timedelta, timezone
from random import randint
import warnings

import faker
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, insert, select

from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import _empty_dag, _fetch_commit_history_edges
from athenian.api.controllers.miners.github.dag_accelerated import join_dags
from athenian.api.controllers.miners.types import nonemin, PullRequestFacts
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.models.metadata.github import Branch
from athenian.api.models.state.models import JIRAProjectSetting
from athenian.api.typing_utils import wraps


@pytest.fixture(scope="function")
def no_deprecation_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture(scope="module")
def release_match_setting_tag_or_branch():
    return {
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags=".*", match=ReleaseMatch.tag_or_branch),
    }


@pytest.fixture(scope="module")
def release_match_setting_tag():
    return {
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches="", tags=".*", match=ReleaseMatch.tag),
    }


@pytest.fixture(scope="module")
def release_match_setting_branch():
    return {
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches=default_branch_alias, tags=".*", match=ReleaseMatch.branch),
    }


@pytest.fixture(scope="module")
def default_branches():
    return {"src-d/go-git": "master",
            "src-d/gitbase": "master"}


_branches = None


@pytest.fixture(scope="function")
async def branches(mdb):
    global _branches
    if _branches is None:
        _branches, _ = await extract_branches(["src-d/go-git"], (6366825,), mdb, None)
    return _branches


@pytest.fixture(scope="function")
async def disabled_dev(sdb):
    await sdb.execute(insert(JIRAProjectSetting).values(
        JIRAProjectSetting(account_id=1, key="DEV", enabled=False)
        .create_defaults().explode(with_primary_keys=True)))


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
            approved_at = fake.date_time_between(
                start_date=first_comment_on_first_review + timedelta(days=1),
                end_date=first_comment_on_first_review + timedelta(days=30),
                tzinfo=timezone.utc)
            last_commit = fake.date_time_between(
                start_date=first_comment_on_first_review + timedelta(days=1),
                end_date=approved_at,
                tzinfo=timezone.utc)
            merged_at = fake.date_time_between(
                approved_at, approved_at + timedelta(days=2), tzinfo=timezone.utc)
            closed_at = merged_at
            last_review = fake.date_time_between(approved_at, closed_at, tzinfo=timezone.utc)
            released_at = fake.date_time_between(
                merged_at, merged_at + timedelta(days=30), tzinfo=timezone.utc)
            reviews = np.array([
                fake.date_time_between(created_at, last_review) for _ in range(randint(0, 3))
            ], dtype="datetime64[ns]")
            activity_days = np.unique(np.array([dt.replace(tzinfo=None) for dt in [
                created_at, closed_at, released_at, first_review_request, first_commit,
                last_commit_before_first_review, last_commit,
            ]] + reviews.tolist(), dtype="datetime64[D]").astype("datetime64[ns]"))
            return PullRequestFacts(
                created=pd.Timestamp(created_at),
                first_commit=pd.Timestamp(first_commit or created_at),
                work_began=nonemin(first_commit, created_at),
                last_commit_before_first_review=pd.Timestamp(last_commit_before_first_review),
                last_commit=pd.Timestamp(last_commit),
                merged=pd.Timestamp(merged_at),
                first_comment_on_first_review=pd.Timestamp(first_comment_on_first_review),
                first_review_request=pd.Timestamp(first_review_request),
                first_review_request_exact=first_review_request,
                last_review=pd.Timestamp(last_review),
                reviews=np.array(reviews),
                activity_days=activity_days,
                approved=pd.Timestamp(approved_at),
                released=pd.Timestamp(released_at),
                closed=pd.Timestamp(closed_at),
                size=randint(10, 1000),
                force_push_dropped=False,
                done=pd.Timestamp(released_at),
            )

        return [random_pr() for _ in range(n)]
    return generate


_dag = None


class FakeFacts:
    def with_repository_full_name(self, _):
        return self

    def with_author(self, _):
        return self

    def with_merger(self, _):
        return self

    def with_releaer(self, _):
        return self


async def fetch_dag(mdb, heads=None):
    if heads is None:
        heads = [
            "MDY6Q29tbWl0NDQ3MzkwNDQ6MTdkYmQ4ODY2MTZmODJiZTJhNTljMGQwMmZkOTNkM2Q2OWYyMzkyYw==",
        ]
    edges = await _fetch_commit_history_edges(heads, [], (6366825,), mdb)
    return {"src-d/go-git": join_dags(*_empty_dag(), edges)}


def with_only_master_branch(func):
    async def wrapped_with_only_master_branch(**kwargs):
        mdb = kwargs["mdb"]
        branches = await mdb.fetch_all(select([Branch]).where(Branch.branch_name != "master"))
        await mdb.execute(delete(Branch).where(Branch.branch_name != "master"))
        try:
            await func(**kwargs)
        finally:
            for branch in branches:
                await mdb.execute(insert(Branch).values(branch))

    return wraps(wrapped_with_only_master_branch, func)


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
