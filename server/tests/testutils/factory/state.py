"""Test factories for models in state DB."""
from datetime import datetime, timedelta, timezone
from typing import Any

import factory

from athenian.api.controllers.invitation_controller import _generate_account_secret
from athenian.api.models.state.models import (
    Account,
    AccountFeature,
    AccountGitHubAccount,
    AccountJiraInstallation,
    Feature,
    FeatureComponent,
    Goal,
    GoalTemplate,
    LogicalRepository,
    MappedJIRAIdentity,
    ReleaseSetting,
    RepositorySet,
    Team,
    TeamGoal,
    UserAccount,
    UserToken,
)
from athenian.api.models.web import PullRequestMetricID

from .alchemy import SQLAlchemyModelFactory
from .common import DEFAULT_ACCOUNT_ID


class LogicalRepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LogicalRepository

    id = factory.Sequence(lambda n: n + 1)
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


class AccountJiraInstallationFactory(SQLAlchemyModelFactory):
    class Meta:
        model = AccountJiraInstallation

    id = factory.Sequence(lambda n: n + 3)
    account_id = factory.Sequence(lambda n: n + 3)


class ReleaseSettingFactory(SQLAlchemyModelFactory):
    class Meta:
        model = ReleaseSetting

    repo_id = factory.Sequence(lambda n: n + 1)
    logical_name = ""
    account_id = DEFAULT_ACCOUNT_ID


class FeatureFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Feature

    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    id = factory.Sequence(lambda n: n + 1)
    name = factory.LazyAttribute(lambda ft: f"feature-{ft.id}")
    component = FeatureComponent.server


class AccountFeatureFactory(SQLAlchemyModelFactory):
    class Meta:
        model = AccountFeature

    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    account_id = DEFAULT_ACCOUNT_ID
    feature_id = factory.Sequence(lambda n: n + 1)


class UserTokenFactory(SQLAlchemyModelFactory):
    class Meta:
        model = UserToken

    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    id = factory.Sequence(lambda n: n + 1)
    account_id = DEFAULT_ACCOUNT_ID
    user_id = "user-00"
    name = factory.LazyAttribute(lambda token: f"token-{token.id}")


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
    name = factory.LazyAttribute(lambda goal: f"goal-{goal.id}")
    metric = PullRequestMetricID.PR_CLOSED
    repositories = None
    jira_projects = None
    jira_priorities = None
    jira_issue_types = None
    metric_params = None
    # create unique intervals by default to avoid uniqueness constraint
    valid_from = factory.Sequence(
        lambda n: datetime(2022, 1, 1).replace(tzinfo=timezone.utc) + timedelta(hours=n),
    )
    expires_at = factory.Sequence(
        lambda n: datetime(2022, 4, 1).replace(tzinfo=timezone.utc) + timedelta(hours=n),
    )


class GoalTemplateFactory(SQLAlchemyModelFactory):
    class Meta:
        model = GoalTemplate

    id = factory.Sequence(lambda n: n + 100)
    account_id = DEFAULT_ACCOUNT_ID
    name = factory.LazyAttribute(lambda template: f"Goal Template {template.id}")
    metric = PullRequestMetricID.PR_DONE
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=50))


class TeamGoalFactory(SQLAlchemyModelFactory):
    class Meta:
        model = TeamGoal

    target = 0
    repositories = None
    jira_projects = None
    jira_priorities = None
    jira_issue_types = None
    metric_params = None


class MappedJIRAIdentityFactory(SQLAlchemyModelFactory):
    class Meta:
        model = MappedJIRAIdentity

    account_id = DEFAULT_ACCOUNT_ID
    github_user_id = factory.Sequence(lambda n: n + 1)
    jira_user_id = factory.LazyAttribute(lambda jira_ident: f"jira-{jira_ident.github_user_id}")
    confidence = 1
