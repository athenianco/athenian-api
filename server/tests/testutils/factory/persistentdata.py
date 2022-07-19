"""Test factories for models in persistentdata DB."""

from datetime import datetime, timedelta, timezone

import factory

from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeploymentNotification,
    ReleaseNotification,
)

from .alchemy import SQLAlchemyModelFactory
from .common import DEFAULT_ACCOUNT_ID


class DeploymentNotificationFactory(SQLAlchemyModelFactory):
    class Meta:
        model = DeploymentNotification

    account_id = DEFAULT_ACCOUNT_ID
    name = factory.Sequence(lambda n: f"deploy {n}")
    conclusion = "SUCCESS"
    environment = "production"
    started_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    finished_at = factory.LazyAttribute(lambda obj: obj.started_at + timedelta(minutes=20))
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))


class DeployedComponentFactory(SQLAlchemyModelFactory):
    class Meta:
        model = DeployedComponent

    account_id = DEFAULT_ACCOUNT_ID
    deployment_name = factory.Sequence(lambda n: f"deploy {n}")
    repository_node_id = 40550  # magic number from mdb
    repository_full_name = "org.org/repo"
    reference = "ABCDEF00"
    resolved_commit_node_id = 2756224  # magic number from mdb


class ReleaseNotificationFactory(SQLAlchemyModelFactory):
    class Meta:
        model = ReleaseNotification

    account_id = DEFAULT_ACCOUNT_ID
    repository_node_id = 1
    commit_hash_prefix = "AAAAAAA"
    published_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
