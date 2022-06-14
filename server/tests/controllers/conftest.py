from datetime import datetime, timedelta, timezone
import logging
from random import randint
from typing import Set
import warnings

import faker
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api import metadata
from athenian.api.defer import with_defer
from athenian.api.internal.features.entries import MetricEntriesCalculator
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github import deployment_light
from athenian.api.internal.miners.github.commit import _empty_dag, _fetch_commit_history_edges
from athenian.api.internal.miners.github.dag_accelerated import join_dags
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.types import PullRequestFacts, nonemin
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    default_branch_alias,
)
from athenian.api.models.metadata.github import Branch
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)
from athenian.api.models.state.models import (
    AccountJiraInstallation,
    JIRAProjectSetting,
    LogicalRepository,
    MappedJIRAIdentity,
    ReleaseSetting,
    RepositorySet,
    Team,
)
from athenian.api.typing_utils import wraps


@pytest.fixture(scope="function")
def no_deprecation_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture(scope="session")
def release_match_setting_tag_or_branch():
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="master", tags=".*", events=".*", match=ReleaseMatch.tag_or_branch,
            ),
        },
    )


@pytest.fixture(scope="session")
def release_match_setting_tag():
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="", tags=".*", events=".*", match=ReleaseMatch.tag,
            ),
        },
    )


@pytest.fixture(scope="session")
def release_match_setting_branch():
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches=default_branch_alias, tags=".*", events=".*", match=ReleaseMatch.branch,
            ),
        },
    )


@pytest.fixture(scope="session")
def release_match_setting_event():
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="", tags=".*", events=".*", match=ReleaseMatch.event,
            ),
        },
    )


@pytest.fixture(scope="session")
def default_branches():
    return {"src-d/go-git": "master", "src-d/gitbase": "master", "src-d/hercules": "master"}


_branches = None


@pytest.fixture(scope="function")
async def branches(mdb, branch_miner, prefixer):
    global _branches
    if _branches is None:
        _branches, _ = await branch_miner.extract_branches(
            ["src-d/go-git"], prefixer, (6366825,), mdb, None,
        )
    return _branches


@pytest.fixture(scope="function")
@with_defer
async def prefixer(mdb):
    return await Prefixer.load((6366825,), mdb, None)


@pytest.fixture(scope="function")
async def disabled_dev(sdb):
    await sdb.execute(
        insert(JIRAProjectSetting).values(
            JIRAProjectSetting(account_id=1, key="DEV", enabled=False)
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def dummy_deployment_label(rdb):
    await rdb.execute(
        insert(DeployedLabel).values(
            DeployedLabel(
                account_id=1,
                deployment_name="Dummy deployment",
                key="xxx",
                value=["yyy"],
            ).explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="session")
def logical_settings():
    return LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"title": ".*[Ff]ix"},
            "src-d/go-git/beta": {"title": ".*[Aa]dd"},
        },
        {},
    )


@pytest.fixture(scope="session")
def logical_settings_labels():
    return LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {
                "labels": ["enhancement", "performance", "plumbing", "ssh", "documentation"],
            },
            "src-d/go-git/beta": {"labels": ["bug", "windows"]},
        },
        {},
    )


@pytest.fixture(scope="session")
def logical_settings_full():
    return LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"title": ".*[Ff]ix"},
            "src-d/go-git/beta": {"title": ".*[Aa]dd"},
        },
        {
            "src-d/go-git/alpha": {"title": ".*(2016|2019)"},
            "src-d/go-git/beta": {"title": "prod|.*2019"},
        },
    )


@pytest.fixture(scope="function")
async def logical_settings_db(sdb):
    await sdb.execute(
        insert(LogicalRepository).values(
            LogicalRepository(
                account_id=1,
                name="alpha",
                repository_id=40550,
                prs={"title": ".*[Ff]ix"},
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(LogicalRepository).values(
            LogicalRepository(
                account_id=1,
                name="beta",
                repository_id=40550,
                prs={"title": ".*[Aa]dd"},
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        update(RepositorySet)
        .where(RepositorySet.owner_id == 1)
        .values(
            {
                RepositorySet.items: [
                    ["github.com/src-d/gitbase", 39652769],
                    ["github.com/src-d/go-git", 40550],
                    ["github.com/src-d/go-git/alpha", 40550],
                    ["github.com/src-d/go-git/beta", 40550],
                ],
                RepositorySet.updates_count: RepositorySet.updates_count + 1,
                RepositorySet.updated_at: datetime.now(timezone.utc),
            },
        ),
    )


@pytest.fixture(scope="session")
def release_match_setting_tag_logical(release_match_setting_tag):
    return ReleaseSettings(
        {
            **release_match_setting_tag.prefixed,
            "github.com/src-d/go-git/alpha": ReleaseMatchSetting(
                branches="", tags=".*", events=".*", match=ReleaseMatch.tag,
            ),
            "github.com/src-d/go-git/beta": ReleaseMatchSetting(
                branches="", tags=r"v4\..*", events=".*", match=ReleaseMatch.tag,
            ),
        },
    )


@pytest.fixture(scope="function")
async def release_match_setting_tag_logical_db(sdb):
    await sdb.execute(
        insert(ReleaseSetting).values(
            ReleaseSetting(
                repository="github.com/src-d/go-git/alpha",
                account_id=1,
                branches="master",
                tags=".*",
                events=".*",
                match=ReleaseMatch.tag,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    await sdb.execute(
        insert(ReleaseSetting).values(
            ReleaseSetting(
                repository="github.com/src-d/go-git/beta",
                account_id=1,
                branches="master",
                tags=r"v4\..*",
                events=".*",
                match=ReleaseMatch.tag,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def logical_reposet(sdb):
    await sdb.execute(
        update(RepositorySet)
        .where(RepositorySet.id == 1)
        .values(
            {
                RepositorySet.items: [
                    ["github.com/src-d/gitbase", 39652769],
                    ["github.com/src-d/go-git", 40550],
                    ["github.com/src-d/go-git/alpha", 40550],
                    ["github.com/src-d/go-git/beta", 40550],
                ],
                RepositorySet.updated_at: datetime.now(timezone.utc),
                RepositorySet.updates_count: 2,
            },
        ),
    )


@pytest.fixture(scope="session")
def pr_samples():
    def generate(n):
        fake = faker.Faker()

        def random_pr():
            created_at = fake.date_time_between(
                start_date="-3y", end_date="-6M", tzinfo=timezone.utc,
            )
            first_commit = fake.date_time_between(
                start_date="-3y1M", end_date=created_at, tzinfo=timezone.utc,
            )
            last_commit_before_first_review = fake.date_time_between(
                start_date=created_at,
                end_date=created_at + timedelta(days=30),
                tzinfo=timezone.utc,
            )
            first_comment_on_first_review = fake.date_time_between(
                start_date=last_commit_before_first_review,
                end_date=timedelta(days=2),
                tzinfo=timezone.utc,
            )
            first_review_request = fake.date_time_between(
                start_date=last_commit_before_first_review,
                end_date=first_comment_on_first_review,
                tzinfo=timezone.utc,
            )
            approved_at = fake.date_time_between(
                start_date=first_comment_on_first_review + timedelta(days=1),
                end_date=first_comment_on_first_review + timedelta(days=30),
                tzinfo=timezone.utc,
            )
            last_commit = fake.date_time_between(
                start_date=first_comment_on_first_review + timedelta(days=1),
                end_date=approved_at,
                tzinfo=timezone.utc,
            )
            merged_at = fake.date_time_between(
                approved_at, approved_at + timedelta(days=2), tzinfo=timezone.utc,
            )
            closed_at = merged_at
            last_review = fake.date_time_between(approved_at, closed_at, tzinfo=timezone.utc)
            released_at = fake.date_time_between(
                merged_at, merged_at + timedelta(days=30), tzinfo=timezone.utc,
            )
            reviews = np.array(
                [fake.date_time_between(created_at, last_review) for _ in range(randint(0, 3))],
                dtype="datetime64[ns]",
            )
            activity_days = np.unique(
                np.array(
                    [
                        dt.replace(tzinfo=None)
                        for dt in [
                            created_at,
                            closed_at,
                            released_at,
                            first_review_request,
                            first_commit,
                            last_commit_before_first_review,
                            last_commit,
                        ]
                    ]
                    + reviews.tolist(),
                    dtype="datetime64[D]",
                ).astype("datetime64[ns]"),
            )
            return PullRequestFacts.from_fields(
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
                release_ignored=False,
                done=pd.Timestamp(released_at),
                review_comments=max(0, randint(-5, 15)),
                regular_comments=max(0, randint(-5, 15)),
                participants=max(randint(-1, 4), 1),
                merged_with_failed_check_runs=["flake8"] if fake.random.random() > 0.9 else [],
            )

        return [random_pr() for _ in range(n)]

    return generate


@pytest.fixture(scope="function")
async def vadim_id_mapping(sdb):
    await sdb.execute(
        insert(MappedJIRAIdentity).values(
            MappedJIRAIdentity(
                account_id=1,
                github_user_id=40020,
                jira_user_id="5de5049e2c5dd20d0f9040c1",
                confidence=1.0,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def denys_id_mapping(sdb):
    await sdb.execute(
        insert(MappedJIRAIdentity).values(
            MappedJIRAIdentity(
                account_id=1,
                github_user_id=40294,
                jira_user_id="5de4cff936b8050e29258600",
                confidence=1.0,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
async def no_jira(sdb):
    await sdb.execute(delete(AccountJiraInstallation))


@pytest.fixture(scope="session")
def bots() -> Set[str]:
    return {
        "login",
        "similar-code-searcher",
        "prettierci",
        "pull",
        "dependabot",
        "changeset-bot",
        "jira",
        "depfu",
        "codecov-io",
        "linear-app",
        "pull-assistant",
        "stale",
        "codecov",
        "sentry-io",
        "minimum-review-bot",
        "sonarcloud",
        "thehub-integration",
        "release-drafter",
        "netlify",
        "height",
        "allcontributors",
        "linc",
        "cla-checker-service",
        "unfurl-links",
        "probot-auto-merge",
        "snyk-bot",
        "slash-commands",
        "greenkeeper",
        "cypress",
        "gally-bot",
        "commitlint",
        "monocodus",
        "dependabot-preview",
        "vercel",
        "codecov-commenter",
        "botelastic",
        "renovate",
        "markdownify",
        "coveralls",
        "github-actions",
        "codeclimate",
        "zube",
    }


_dag = None


class FakeFacts(PullRequestFacts):
    def __init__(self):
        super().__init__(b"\0" * PullRequestFacts.dtype.itemsize)


async def fetch_dag(mdb, heads=None):
    if heads is None:
        heads = [
            2755363,
        ]
    edges = await _fetch_commit_history_edges(heads, [], (6366825,), mdb)
    return {"src-d/go-git": join_dags(*_empty_dag(), edges)}


def with_only_master_branch(func):
    async def wrapped_with_only_master_branch(**kwargs):
        mdb = kwargs["mdb_rw"]
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


@pytest.fixture(scope="function")
async def metrics_calculator_factory(mdb, pdb, rdb, cache):
    def build(account_id, meta_ids, with_cache=False, cache_only=False):
        if cache_only:
            return MetricEntriesCalculator(account_id, meta_ids, 28, None, None, None, cache)
        if with_cache:
            c = cache
        else:
            c = None

        return MetricEntriesCalculator(account_id, meta_ids, 28, mdb, pdb, rdb, c)

    return build


@pytest.fixture(scope="function")
async def metrics_calculator_factory_memcached(mdb, pdb, rdb, memcached):
    def build(account_id, meta_ids, with_cache=False, cache_only=False):
        if cache_only:
            return MetricEntriesCalculator(account_id, meta_ids, 28, None, None, None, memcached)
        if with_cache:
            c = memcached
        else:
            c = None

        return MetricEntriesCalculator(account_id, meta_ids, 28, mdb, pdb, rdb, c)

    return build


@pytest.fixture(scope="function")
@with_defer
async def precomputed_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
):
    await _precompute_deployments(
        release_match_setting_tag_or_branch, prefixer, branches, default_branches, mdb, pdb, rdb,
    )


@pytest.fixture(scope="function")
async def sample_deployments(rdb):
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    for year, month, day, conclusion, tag, commit in (
        (2019, 11, 1, "SUCCESS", "v4.13.1", 2755244),
        (2018, 12, 1, "SUCCESS", "4.8.1", 2755046),
        (2018, 12, 2, "SUCCESS", "4.8.1", 2755046),
        (2018, 8, 1, "SUCCESS", "4.5.0", 2755028),
        (2016, 12, 1, "SUCCESS", "3.2.0", 2755108),
        (2018, 1, 12, "FAILURE", "4.0.0", 2757510),
        (2018, 1, 11, "SUCCESS", "4.0.0", 2757510),
        (2018, 1, 10, "FAILURE", "4.0.0", 2757510),
        (2016, 7, 6, "SUCCESS", "3.1.0", 2756224),
    ):
        for env in ("production", "staging", "canary"):
            name = "%s_%d_%02d_%02d" % (env, year, month, day)
            await rdb.execute(
                insert(DeploymentNotification).values(
                    dict(
                        account_id=1,
                        name=name,
                        conclusion=conclusion,
                        environment=env,
                        started_at=datetime(year, month, day, tzinfo=timezone.utc),
                        finished_at=datetime(year, month, day, 0, 10, tzinfo=timezone.utc),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    ),
                ),
            )
            await rdb.execute(
                insert(DeployedComponent).values(
                    dict(
                        account_id=1,
                        deployment_name=name,
                        repository_node_id=40550,
                        reference=tag,
                        resolved_commit_node_id=commit,
                        created_at=datetime.now(timezone.utc),
                    ),
                ),
            )


@pytest.fixture(scope="function")
@with_defer
async def precomputed_sample_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    sample_deployments,
):
    await _precompute_deployments(
        release_match_setting_tag_or_branch, prefixer, branches, default_branches, mdb, pdb, rdb,
    )


async def _precompute_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
):
    deps, _ = await mine_deployments(
        ["src-d/go-git"],
        {},
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        ["production", "staging"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches,
        default_branches,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    log = logging.getLogger(f"{metadata.__package__}.precomputed_deployments")
    log.info("Mined %d deployments", len(deps))
    log.info("Mined %d release deployments", sum(len(df) for df in deps["releases"].values))


@pytest.fixture(scope="function")
def detect_deployments(request):
    repository_environment_threshold = deployment_light.repository_environment_threshold
    deployment_light.repository_environment_threshold = timedelta(days=100 * 365)

    def restore_repository_environment_threshold():
        deployment_light.repository_environment_threshold = repository_environment_threshold

    request.addfinalizer(restore_repository_environment_threshold)


def pytest_configure(config):
    for mark in (
        "filter_repositories",
        "filter_contributors",
        "filter_pull_requests",
        "filter_commits",
        "filter_releases",
        "filter_labels",
    ):
        config.addinivalue_line("markers", mark)


@pytest.fixture(scope="function")
async def sample_team(sdb):
    return await sdb.execute(
        insert(Team).values(
            Team(
                owner_id=1,
                name="Sample",
                members=[51, 40020, 39789, 40070],
                parent_id=None,
            )
            .create_defaults()
            .explode(),
        ),
    )
