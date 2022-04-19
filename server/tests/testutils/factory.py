"""Test factories for models used by application.

Factories are implemented using factory_boy library.
"""

from datetime import datetime, timedelta, timezone
import hashlib

import factory

from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubOpenPullRequestFacts, GitHubRelease
from athenian.api.models.state.models import LogicalRepository
from tests.testutils.factory_alchemy import SQLAlchemyModelFactory

_DEFAULT_ACCOUNT_ID = 1


class GitHubDonePullRequestFactsFactory(SQLAlchemyModelFactory):
    class Meta:
        model = GitHubDonePullRequestFacts

    acc_id = _DEFAULT_ACCOUNT_ID
    pr_node_id = factory.Sequence(lambda n: n)
    release_match = "branch|master"
    repository_full_name = factory.Sequence(lambda n: f"athenianco/repo-{n}")
    pr_created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
    pr_done_at = factory.LazyAttribute(lambda o: o.pr_created_at + timedelta(hours=10))
    number = factory.Sequence(lambda n: n)
    data = b""


class GitHubOpenPullRequestFactsFactory(SQLAlchemyModelFactory):
    class Meta:
        model = GitHubOpenPullRequestFacts

    acc_id = _DEFAULT_ACCOUNT_ID
    pr_node_id = factory.Sequence(lambda n: n)
    repository_full_name = factory.Sequence(lambda n: f"athenianco/repo-{n}")
    pr_created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
    number = factory.Sequence(lambda n: n)
    pr_updated_at = factory.LazyAttribute(lambda o: o.pr_created_at + timedelta(hours=6))
    data = b""


class LogicalRepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LogicalRepository

    account_id = _DEFAULT_ACCOUNT_ID
    name = factory.Sequence(lambda n: f"logical-repo-{n}")
    repository_id = factory.Sequence(lambda n: n)


class GitHubReleaseFactory(SQLAlchemyModelFactory):
    class Meta:
        model = GitHubRelease

    acc_id = _DEFAULT_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n)
    release_match = "branch|master"
    repository_full_name = factory.Sequence(lambda n: f"athenianco/repo-{n}")
    repository_node_id = factory.Sequence(lambda n: n)
    name = factory.Sequence(lambda n: f"v-{n}")
    published_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    url = factory.Sequence(lambda n: f"https://example.com/releases/v-{n}")
    sha = hashlib.sha1(b"").hexdigest()
    commit_id = factory.Sequence(lambda n: n)
