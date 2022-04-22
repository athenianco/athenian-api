from datetime import datetime, timedelta, timezone

import factory

from athenian.api.models.metadata.github import Account, AccountRepository, FetchProgress

from .alchemy import SQLAlchemyModelFactory


class AccountRepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = AccountRepository

    acc_id = factory.Sequence(lambda n: n)
    repo_graph_id = factory.Sequence(lambda n: n)
    repo_full_name = factory.Sequence(lambda n: f"athenianco/proj-{n:02}")
    event_id = "event-00"
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    enabled = True


class AccountFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Account

    id = factory.Sequence(lambda n: n)
    owner_id = factory.Sequence(lambda n: n)
    owner_login = factory.Sequence(lambda n: f"login-{n}")
    name = factory.Sequence(lambda n: f"name-{n}")


class FetchProgressFactory(SQLAlchemyModelFactory):
    class Meta:
        model = FetchProgress

    acc_id = factory.Sequence(lambda n: n)
    event_id = factory.Sequence(lambda n: n)
    node_type = "Repository"
    nodes_total = 100
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=3))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
