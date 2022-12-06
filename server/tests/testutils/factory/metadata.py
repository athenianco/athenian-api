from datetime import datetime, timedelta, timezone

import factory

from athenian.api.models.metadata.github import (
    Account,
    AccountRepository,
    Bot,
    FetchProgress,
    NodePullRequest,
    NodePullRequestJiraIssues,
    PullRequest,
    PullRequestCommit,
    PullRequestReview,
    PullRequestReviewRequest,
    Repository,
    Team,
    TeamMember,
    User,
)
from athenian.api.models.metadata.jira import (
    AthenianIssue,
    Issue,
    IssueType,
    Priority,
    Progress,
    Project,
    User as JIRAUser,
)

from .alchemy import SQLAlchemyModelFactory
from .common import DEFAULT_JIRA_ACCOUNT_ID, DEFAULT_MD_ACCOUNT_ID


class AccountRepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = AccountRepository

    acc_id = DEFAULT_MD_ACCOUNT_ID
    repo_graph_id = factory.Sequence(lambda n: n)
    repo_full_name = factory.Sequence(lambda n: f"athenianco/proj-{n:02}")
    event_id = "event-00"
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class PullRequestCommitFactory(SQLAlchemyModelFactory):
    class Meta:
        model = PullRequestCommit

    acc_id = DEFAULT_MD_ACCOUNT_ID
    sha = "a" * 40
    node_id = factory.Sequence(lambda n: n + 10000)
    commit_node_id = factory.LazyAttribute(lambda c: c.node_id)
    committed_date = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=3))
    authored_date = factory.LazyAttribute(lambda c: c.committed_date)
    commit_date = factory.LazyAttribute(lambda c: c.committed_date.isoformat())
    author_date = factory.LazyAttribute(lambda c: c.authored_date.isoformat())
    pull_request_node_id = factory.Sequence(lambda n: n + 1)
    repository_full_name = "org/repo"
    repository_node_id = factory.Sequence(lambda n: n + 100)


class PullRequestReviewFactory(SQLAlchemyModelFactory):
    class Meta:
        model = PullRequestReview

    acc_id = DEFAULT_MD_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n + 10000)
    submitted_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    state = "APPROVED"
    pull_request_node_id = factory.Sequence(lambda n: n + 1)
    repository_full_name = "org/repo"
    repository_node_id = factory.Sequence(lambda n: n + 100)


class PullRequestReviewRequestFactory(SQLAlchemyModelFactory):
    class Meta:
        model = PullRequestReviewRequest

    acc_id = DEFAULT_MD_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n + 10000)  # do not step over the Big Fixture
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    pull_request_id = factory.Sequence(lambda n: n + 1)


class PullRequestFactory(SQLAlchemyModelFactory):
    class Meta:
        model = PullRequest

    acc_id = DEFAULT_MD_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n + 1)
    repository_full_name = "org/repo"
    repository_node_id = factory.Sequence(lambda n: n + 100)
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=3))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
    user_node_id = factory.Sequence(lambda n: n + 1000)
    user_login = factory.LazyAttribute(lambda pr: f"user-{pr.user_node_id}")
    additions = 1
    base_ref = "ABCDE"
    changed_files = 1
    deletions = 0
    commits = 1
    head_ref = "01234"
    number = factory.Sequence(lambda n: n + 1)
    title = factory.LazyAttribute(lambda pr: f"PR {pr.number}")


class RepositoryFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Repository

    acc_id = DEFAULT_MD_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n + 1)
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=3))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
    full_name = factory.LazyAttribute(lambda repo: f"org/repo-{repo.node_id}")
    html_url = factory.LazyAttribute(lambda repo: f"https://github.com/{repo.full_name}")


class AccountFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Account

    id = factory.Sequence(lambda n: n)
    owner_id = factory.Sequence(lambda n: n)
    owner_login = factory.Sequence(lambda n: f"login-{n}")
    name = factory.Sequence(lambda n: f"name-{n}")
    install_url = factory.Sequence(
        lambda n: f"https://github.com/organizations/athenianco/settings/installations/{n}",
    )


class FetchProgressFactory(SQLAlchemyModelFactory):
    class Meta:
        model = FetchProgress

    acc_id = factory.Sequence(lambda n: n)
    event_id = factory.Sequence(lambda n: n)
    node_type = "Repository"
    nodes_total = 100
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=3))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))


class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User

    acc_id = DEFAULT_MD_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n + 1)
    avatar_url = factory.LazyAttribute(lambda user: f"https://github.com/user-{user.node_id}.jpg")
    login = factory.LazyAttribute(lambda user: f"user-{user.node_id}")
    html_url = factory.LazyAttribute(lambda user: f"https://github.com/user-{user.node_id}")


class NodePullRequestFactory(SQLAlchemyModelFactory):
    class Meta:
        model = NodePullRequest

    acc_id = DEFAULT_MD_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n + 1)
    title = factory.LazyAttribute(lambda pr: f"PR [{pr.node_id}]")
    additions = 1
    deletions = 1
    number = factory.LazyAttribute(lambda pr: pr.node_id)
    repository_id = 1
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=3))
    updated_at = factory.LazyAttribute(lambda pr: pr.created_at + timedelta(days=2))


class BotFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Bot

    acc_id = DEFAULT_MD_ACCOUNT_ID
    login = factory.LazyAttribute(lambda bot: f"bot-{bot.node_id}")


class NodePullRequestJiraIssuesFactory(SQLAlchemyModelFactory):
    class Meta:
        model = NodePullRequestJiraIssues

    node_id = factory.Sequence(lambda n: n + 1)
    node_acc = DEFAULT_MD_ACCOUNT_ID
    jira_id = factory.Sequence(lambda n: str(n + 100))
    jira_acc = DEFAULT_JIRA_ACCOUNT_ID


class TeamFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Team

    acc_id = DEFAULT_MD_ACCOUNT_ID
    node_id = factory.Sequence(lambda n: n + 1)
    organization_id = 40469
    name = factory.LazyAttribute(lambda team: f"team-{team.node_id}")


class TeamMemberFactory(SQLAlchemyModelFactory):
    class Meta:
        model = TeamMember

    acc_id = DEFAULT_MD_ACCOUNT_ID
    parent_id = factory.Sequence(lambda n: n + 1)
    child_id = factory.Sequence(lambda n: n + 2)


class JIRAAthenianIssueFactory(SQLAlchemyModelFactory):
    class Meta:
        model = AthenianIssue

    acc_id = DEFAULT_JIRA_ACCOUNT_ID
    id = factory.Sequence(lambda n: str(n + 1))
    work_began = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=5))
    resolved = None
    updated = None
    nested_assignee_display_names: dict = {}


class JIRAProjectFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Project

    acc_id = DEFAULT_JIRA_ACCOUNT_ID
    id = factory.Sequence(lambda n: str(n + 1))
    key = factory.Sequence(lambda n: f"PRJ-{n + 1:03d}")
    name = factory.Sequence(lambda n: f"Project {n + 1:03d}")
    avatar_url = factory.Sequence(lambda n: "https://jira.com/proj-{n + 1}/avatar")
    is_deleted = False


class JIRAIssueTypeFactory(SQLAlchemyModelFactory):
    class Meta:
        model = IssueType

    acc_id = DEFAULT_JIRA_ACCOUNT_ID
    id = factory.Sequence(lambda n: str(n + 1))
    project_id = factory.Sequence(lambda n: f"proj-{n + 1}")
    name = factory.LazyAttribute(lambda issue: f"ISSUETYPE-{issue.id}")
    icon_url = factory.LazyAttribute(lambda i: f"https://jira.com/issuetype-{i.id}-icon.png")
    is_subtask = False
    is_epic = False
    normalized_name = factory.LazyAttribute(lambda issue: issue.name.lower())


class JIRAIssueFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Issue

    acc_id = DEFAULT_JIRA_ACCOUNT_ID
    id = factory.Sequence(lambda n: str(n + 1))
    project_id = factory.Sequence(lambda n: f"proj-{n + 1}")
    key = factory.LazyAttribute(lambda issue: f"ISSUE-{issue.id}")
    title = factory.LazyAttribute(lambda issue: f"Issue n. {issue.id}")
    type = "Task"
    type_id = "1"
    labels: list[str] = []
    epic_id = factory.Sequence(lambda n: str(n + 1000))
    created = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=3))
    updated = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
    reporter_id = "reporter-1"
    reporter_display_name = "Reporter N. 1"
    comments_count = 0
    priority_id = "1"
    priority_name = "HIGH"
    status_id = "1"
    status = "todo"
    url = factory.LazyAttribute(lambda issue: f"https://jira.com/issue-{issue.id}")

    @factory.post_generation
    def athenian_epic_id(obj, create, extracted, **kwargs):
        # adding athenian_epic_id attr is required since explode_model() doesn't
        # work with aliased columns, i.e. epic_id = Column("athenian_epic_id", Text)
        obj.athenian_epic_id = obj.epic_id


class JIRAUserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = JIRAUser

    acc_id = DEFAULT_JIRA_ACCOUNT_ID
    id = factory.Sequence(lambda n: f"{n + 1}")
    type = "?"
    display_name = factory.LazyAttribute(lambda user: f"jira user {user.id}")
    avatar_url = factory.LazyAttribute(lambda user: f"https://jira.com/user-{user.id}.jpg")


class JIRAPriorityFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Priority

    acc_id = DEFAULT_JIRA_ACCOUNT_ID
    id = factory.Sequence(lambda n: str(n + 1))
    name = factory.LazyAttribute(lambda issue: f"PRIORITY-{issue.id}")
    rank = 1
    status_color = "#ffffff"


class JIRAProgressFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Progress

    acc_id = DEFAULT_JIRA_ACCOUNT_ID
    event_id = factory.Sequence(lambda n: str(n + 1))
    event_type = "?"
    current = 10
    total = 100
    is_initial = False
    started_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
    end_at = factory.LazyFunction(lambda: datetime.now(timezone.utc) - timedelta(days=1))
