"""Test factories for models in state DB."""
from datetime import datetime, timedelta, timezone
from typing import Any

import factory

from athenian.api.controllers.invitation_controller import _generate_account_secret
from athenian.api.models.state.models import (
    Account,
    AccountGitHubAccount,
    Goal,
    LogicalRepository,
    MappedJIRAIdentity,
    RepositorySet,
    Team,
    TeamGoal,
    UserAccount,
)

from .alchemy import SQLAlchemyModelFactory
from .common import DEFAULT_ACCOUNT_ID


class LogicalRepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LogicalRepository

    account_id = DEFAULT_ACCOUNT_ID
    name = factory.Sequence(lambda n: f"logical-repo-{n}")
    repository_id = factory.Sequence(lambda n: n + 1)


class AccountFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Account

    id = factory.Sequence(lambda n: n + 3)
    expires_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) + timedelta(days=100))

    @factory.post_generation
    def _generate_secret(obj: Account, create: bool, extracted: Any, **kwargs: Any):
        obj.secret_salt, obj.secret = _generate_account_secret(obj.id, "secret")


class AccountGitHubAccountFactory(SQLAlchemyModelFactory):
    class Meta:
        model = AccountGitHubAccount

    id = factory.Sequence(lambda n: n + 3)
    account_id = factory.Sequence(lambda n: n + 3)


class UserAccountFactory(SQLAlchemyModelFactory):
    class Meta:
        model = UserAccount

    user_id = factory.Sequence(lambda n: f"github|{n:05}")
    account_id = DEFAULT_ACCOUNT_ID
    is_admin = True


class RepositorySetFactory(SQLAlchemyModelFactory):
    class Meta:
        model = RepositorySet

    id = factory.Sequence(lambda n: n + 1)
    name = RepositorySet.ALL
    owner_id = DEFAULT_ACCOUNT_ID
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    items = []


class TeamFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Team

    id = factory.Sequence(lambda n: n + 1)
    owner_id = DEFAULT_ACCOUNT_ID
    parent_id = None
    name = factory.LazyAttribute(lambda team: f"team-{team.id:03}")
    members = []


class GoalFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Goal

    id = factory.Sequence(lambda n: n + 1)
    account_id = DEFAULT_ACCOUNT_ID
    template_id = 1
    valid_from = factory.LazyFunction(lambda: datetime(2022, 1, 1).replace(tzinfo=timezone.utc))
    expires_at = factory.LazyFunction(lambda: datetime(2022, 4, 1).replace(tzinfo=timezone.utc))


class TeamGoalFactory(SQLAlchemyModelFactory):
    class Meta:
        model = TeamGoal

    target = 0


class MappedJIRAIdentityFactory(SQLAlchemyModelFactory):
    class Meta:
        model = MappedJIRAIdentity

    account_id = DEFAULT_ACCOUNT_ID
    github_user_id = factory.Sequence(lambda n: n + 1)
    jira_user_id = factory.LazyAttribute(lambda jira_ident: f"jira-{jira_ident.github_user_id}")
    confidence = 1
