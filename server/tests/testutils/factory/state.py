"""Test factories for models in state DB."""


import factory

from athenian.api.models.state.models import LogicalRepository

from .alchemy import SQLAlchemyModelFactory
from .common import DEFAULT_ACCOUNT_ID


class LogicalRepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LogicalRepository

    account_id = DEFAULT_ACCOUNT_ID
    name = factory.Sequence(lambda n: f"logical-repo-{n}")
    repository_id = factory.Sequence(lambda n: n)
