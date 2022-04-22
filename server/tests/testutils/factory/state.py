"""Test factories for models in state DB."""
from datetime import datetime, timedelta, timezone
from typing import Any

import factory

from athenian.api.controllers.invitation_controller import _generate_account_secret
from athenian.api.models.state.models import Account, LogicalRepository, RepositorySet

from .alchemy import SQLAlchemyModelFactory
from .common import DEFAULT_ACCOUNT_ID


class LogicalRepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LogicalRepository

    account_id = DEFAULT_ACCOUNT_ID
    name = factory.Sequence(lambda n: f"logical-repo-{n}")
    repository_id = factory.Sequence(lambda n: n)


class AccountFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Account

    id = factory.Sequence(lambda n: n + 3)
    expires_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) + timedelta(days=100))

    @factory.post_generation
    def _generate_secret(obj: Account, create: bool, extracted: Any, **kwargs: Any):
        obj.secret_salt, obj.secret = _generate_account_secret(obj.id, "secret")


class RepositorySetFactory(SQLAlchemyModelFactory):
    class Meta:
        model = RepositorySet

    id = factory.Sequence(lambda n: n)
    name = RepositorySet.ALL
    owner_id = DEFAULT_ACCOUNT_ID
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    items = []
