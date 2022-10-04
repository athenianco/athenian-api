from datetime import datetime, timedelta, timezone
import logging
import warnings

import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api import metadata
from athenian.api.defer import with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github import deployment_light
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.types import PullRequestFacts
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.metadata.github import Branch
from athenian.api.models.persistentdata.models import DeployedLabel
from athenian.api.models.state.models import (
    AccountJiraInstallation,
    JIRAProjectSetting,
    MappedJIRAIdentity,
    RepositorySet,
    Team,
)
from athenian.api.typing_utils import wraps


@pytest.fixture(scope="function")
def no_deprecation_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


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


class FakeFacts(PullRequestFacts):
    def __init__(self):
        super().__init__(b"\0" * PullRequestFacts.dtype.itemsize)


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
    deps = await mine_deployments(
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
        None,
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
