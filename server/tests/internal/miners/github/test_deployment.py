from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
import pickle
from typing import Any

import medvedi as md
from medvedi.testing import assert_frame_equal, assert_index_equal
import morcilla
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import sqlalchemy as sa
from sqlalchemy import delete, func, insert, select

from athenian.api.async_utils import gather
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.dag_accelerated import (
    extract_independent_ownership,
    extract_pr_commits,
)
from athenian.api.internal.miners.github.deployment import (
    MineDeploymentsMetrics,
    deployment_facts_extract_mentioned_people,
    hide_outlier_first_deployments,
    invalidate_precomputed_on_labels_change,
    mine_deployments,
    reset_broken_deployments,
)
from athenian.api.internal.miners.github.release_mine import mine_releases
from athenian.api.internal.miners.types import DeploymentConclusion, DeploymentFacts, ReleaseFacts
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, Settings
from athenian.api.models.metadata.github import Release
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
    ReleaseNotification,
)
from athenian.api.models.precomputed.models import (
    GitHubCommitDeployment,
    GitHubDeploymentFacts,
    GitHubPullRequestDeployment,
    GitHubReleaseDeployment,
)
from tests.testutils.db import (
    Database,
    DBCleaner,
    assert_existing_row,
    assert_missing_row,
    count,
    models_insert,
)
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.persistentdata import (
    DeployedComponentFactory,
    DeployedLabelFactory,
    DeploymentNotificationFactory,
)
from tests.testutils.factory.precomputed import GitHubDeploymentFactsFactory
from tests.testutils.factory.wizards import insert_logical_repo, insert_repo
from tests.testutils.time import dt


@pytest.mark.parametrize("eager_filter_repositories", [False, True])
@with_defer
async def test_mine_deployments_from_scratch(
    sample_deployments,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
    eager_filter_repositories,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    metrics = MineDeploymentsMetrics.empty()
    await mine_releases(
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_avatars=False,
        with_extended_pr_details=False,
        with_deployments=False,
    )
    await wait_deferred()

    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        cache,
        metrics=metrics,
        eager_filter_repositories=eager_filter_repositories,
    )
    _validate_deployments(
        deps, 7 if eager_filter_repositories else 9, True, eager_filter_repositories,
    )
    deployment_facts_extract_mentioned_people(deps)
    await wait_deferred()
    commits = await pdb.fetch_all(select(GitHubCommitDeployment))
    assert len(commits) == 4684
    assert metrics.count == 14 if eager_filter_repositories else 18
    assert metrics.unresolved == 0

    # test the cache
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        None,
        None,
        None,
        cache,
        eager_filter_repositories=eager_filter_repositories,
    )
    _validate_deployments(
        deps, 7 if eager_filter_repositories else 9, True, eager_filter_repositories,
    )


@with_defer
async def test_mine_deployments_unresolved(
    sample_deployments,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    metrics = MineDeploymentsMetrics.empty()

    await rdb.execute(
        insert(DeploymentNotification).values(
            account_id=1,
            name="whatever",
            conclusion="SUCCESS",
            environment="production",
            started_at=datetime(2019, 11, 2, tzinfo=timezone.utc),
            finished_at=datetime(2019, 11, 2, 0, 10, tzinfo=timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    )
    await rdb.execute(
        insert(DeployedComponent).values(
            account_id=1,
            deployment_name="whatever",
            repository_node_id=40550,
            reference="not exists",
            created_at=datetime.now(timezone.utc),
        ),
    )

    await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        cache,
        eager_filter_repositories=False,
        metrics=metrics,
    )
    assert metrics.count == 18
    assert metrics.unresolved == 1


@with_defer
async def test_mine_deployments_addons_cache(
    sample_deployments,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_avatars=False,
        with_extended_pr_details=False,
        with_deployments=False,
    )
    await wait_deferred()
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        JIRAConfig(1, {}, {}),
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
        with_extended_prs=True,
        with_jira=True,
        eager_filter_repositories=False,
    )
    _validate_deployments(deps, 9, True, False)
    deployment_facts_extract_mentioned_people(deps)
    await wait_deferred()

    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        JIRAConfig(1, {}, {}),
        (6366825,),
        None,
        None,
        None,
        cache,
        eager_filter_repositories=False,
    )
    _validate_deployments(deps, 9, True, False)


@with_defer
async def test_mine_deployments_middle(
    sample_deployments,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    time_from = datetime(2017, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        datetime(2016, 1, 1, tzinfo=timezone.utc),
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_avatars=False,
        with_extended_pr_details=False,
        with_deployments=False,
    )
    await wait_deferred()
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        cache,
        eager_filter_repositories=False,
    )
    _validate_deployments(deps, 7, False, False)
    deployment_facts_extract_mentioned_people(deps)


@with_defer
async def test_mine_deployments_append(
    sample_deployments,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 11, 2, tzinfo=timezone.utc)
    await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        cache,
        eager_filter_repositories=False,
    )
    await wait_deferred()
    name = "%s_%d_%02d_%02d" % ("production", 2019, 11, 2)
    await rdb.execute(
        insert(DeploymentNotification).values(
            account_id=1,
            name=name,
            conclusion="SUCCESS",
            environment="production",
            started_at=datetime(2019, 11, 2, tzinfo=timezone.utc),
            finished_at=datetime(2019, 11, 2, 0, 10, tzinfo=timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    )
    await rdb.execute(
        insert(DeployedComponent).values(
            account_id=1,
            deployment_name=name,
            repository_node_id=40550,
            reference="v4.13.1",
            resolved_commit_node_id=2755244,
            created_at=datetime.now(timezone.utc),
        ),
    )
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to + timedelta(days=1),
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
        cache,
        eager_filter_repositories=False,
    )
    await wait_deferred()
    i = np.argmax(deps.index.get_level_values(0) == name)
    assert len(deps["prs"][i]) == 0
    assert len(deps["releases"][i]) == 0
    await _validate_deployed_prs(pdb)


@with_defer
async def test_mine_deployments_insert_middle(
    sample_deployments,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 12, 31, tzinfo=timezone.utc)
    await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production"],
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
        cache,
    )
    await wait_deferred()
    time_from = datetime(2015, 12, 31, tzinfo=timezone.utc)
    time_to = datetime(2019, 12, 31, tzinfo=timezone.utc)
    await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production"],
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
        cache,
    )
    await wait_deferred()
    await _validate_deployed_prs(pdb)


@with_defer
async def test_mine_deployments_only_failed(
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    for year, month, day, conclusion, tag, commit in ((2018, 1, 10, "FAILURE", "4.0.0", 2757510),):
        name = "production_%d_%02d_%02d" % (year, month, day)
        await rdb.execute(
            insert(DeploymentNotification).values(
                account_id=1,
                name=name,
                conclusion=conclusion,
                environment="production",
                started_at=datetime(year, month, day, tzinfo=timezone.utc),
                finished_at=datetime(year, month, day, 0, 10, tzinfo=timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        )
        await rdb.execute(
            insert(DeployedComponent).values(
                account_id=1,
                deployment_name=name,
                repository_node_id=40550,
                reference=tag,
                resolved_commit_node_id=commit,
                created_at=datetime.now(timezone.utc),
            ),
        )
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 11, 2, tzinfo=timezone.utc)
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        cache,
    )
    await wait_deferred()
    assert len(deps) == 1
    assert len(deps.iloc[0]["prs"]) == 340
    rows = await pdb.fetch_all(select(GitHubPullRequestDeployment))
    assert len(rows) == 340


@pytest.mark.parametrize("eager_filter_repositories", [False, True])
@with_defer
async def test_mine_deployments_logical(
    sample_deployments,
    release_match_setting_tag_logical,
    branches,
    default_branches,
    prefixer,
    logical_settings_full,
    mdb,
    pdb,
    rdb,
    cache,
    eager_filter_repositories,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git/alpha", "src-d/go-git/beta", "src-d/go-git"],
        {},
        branches,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_logical,
        logical_settings_full,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_avatars=False,
        with_extended_pr_details=False,
        with_deployments=False,
    )
    await wait_deferred()
    deps = await mine_deployments(
        ["src-d/go-git/alpha"],
        {},
        time_from,
        time_to,
        ["production", "staging"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_logical,
        logical_settings_full,
        branches,
        default_branches,
        prefixer,
        1,
        None,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
        eager_filter_repositories=eager_filter_repositories,
    )
    assert len(deps) == 6 if eager_filter_repositories else 18
    physical_count = alpha_count = beta_count = beta_releases = 0
    for deployment_name, components, releases in zip(
        deps.index.values,
        deps["components"],
        deps["releases"],
    ):
        component_repos = components.unique(DeployedComponent.repository_full_name)
        release_repos = (
            np.unique(releases.index.get_level_values(1))
            if not releases.empty
            else np.array([], dtype=object)
        )
        has_logical = False
        if "2016" in deployment_name or "2019" in deployment_name:
            has_logical = True
            alpha_count += 1
            assert "src-d/go-git/alpha" in component_repos, deployment_name
            assert "src-d/go-git" not in component_repos, deployment_name
            assert "src-d/go-git/alpha" in release_repos, deployment_name
            assert "src-d/go-git" not in release_repos, deployment_name
        if "prod" in deployment_name or "2019" in deployment_name:
            has_logical = True
            beta_count += 1
            assert "src-d/go-git/beta" in component_repos, deployment_name
            assert "src-d/go-git" not in component_repos, deployment_name
            beta_releases += "src-d/go-git/beta" in release_repos
            assert "src-d/go-git" not in release_repos, deployment_name
        if not has_logical:
            physical_count += 1
            assert component_repos.tolist() == ["src-d/go-git"]
            assert release_repos.tolist() in ([], ["src-d/go-git"])

    assert alpha_count == 6
    assert beta_count == 4 if eager_filter_repositories else 10
    assert physical_count == 0 if eager_filter_repositories else 6
    assert beta_releases == 2 if eager_filter_repositories else 6


@with_defer
async def test_mine_deployments_no_prs(
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2016, 1, 1, tzinfo=timezone.utc)
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    await rdb.execute(
        insert(DeploymentNotification).values(
            account_id=1,
            name="DeployWithoutPRs",
            conclusion="SUCCESS",
            environment="production",
            started_at=datetime(2015, 5, 21, tzinfo=timezone.utc),
            finished_at=datetime(2015, 5, 21, 0, 10, tzinfo=timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    )
    await rdb.execute(
        insert(DeployedComponent).values(
            account_id=1,
            deployment_name="DeployWithoutPRs",
            repository_node_id=40550,
            reference="35b585759cbf29f8ec428ef89da20705d59f99ec",
            resolved_commit_node_id=2755715,
            created_at=datetime.now(timezone.utc),
        ),
    )
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production"],
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
        cache,
    )
    assert len(deps) == 1
    assert len(deps.iloc[0]["prs"]) == 0


@with_defer
async def test_mine_deployments_no_release_facts(
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    from .mine_deployments_data import GROUND_TRUTH_0

    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        cache,
    )
    assert len(deps) == 1
    assert deps.iloc[0][DeploymentFacts.f.name] == "Dummy deployment"
    deployment_facts_extract_mentioned_people(deps)
    dfs = deps["releases"][0]
    obj = {
        c: {
            (node_id, repo): v if not isinstance(v, np.ndarray) else v.tolist()
            for node_id, repo, v in zip(*dfs.index.levels(), dfs[c])
        }
        for c in dfs.columns
    }
    for col in (
        ReleaseFacts.f.prs_created_at,
        ReleaseFacts.f.node_id,
        ReleaseFacts.f.repository_full_name,
    ):
        del obj[col]

    all_keys = list(obj[ReleaseFacts.f.commit_authors])
    for field in (
        "jira_ids",
        "jira_pr_offsets",
        "jira_priorities",
        "jira_projects",
        "jira_types",
        "jira_labels",
    ):
        assert obj.pop(field) == {k: [] for k in all_keys}

    for field in (ReleaseFacts.f.prs_title, ReleaseFacts.f.deployments):
        assert obj.pop(field) == {k: None for k in all_keys}

    for topic, values in GROUND_TRUTH_0.items():
        assert obj[topic] == values, topic
    diff_keys = obj.keys() - GROUND_TRUTH_0.keys()
    assert not diff_keys


@with_defer
async def test_mine_deployments_precomputed_dummy(
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps1 = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
    people1 = deployment_facts_extract_mentioned_people(deps1)
    await wait_deferred()
    deps2 = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
    people2 = deployment_facts_extract_mentioned_people(deps2)
    assert len(deps1) == len(deps2) == 1
    assert_index_equal(deps1.index, deps2.index)
    for topic in ("releases", "components", "labels"):
        for pdf, sdf in zip(deps1[topic], deps2[topic]):
            assert_frame_equal(pdf, sdf)
    for df in (deps1, deps2):
        for col in ("releases", "components", "labels"):
            del df[col]
    assert_frame_equal(deps1, deps2)
    assert (people1 == people2).all()


@with_defer
async def test_mine_deployments_precomputed_sample(
    sample_deployments,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps1 = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        eager_filter_repositories=False,
    )
    people1 = deployment_facts_extract_mentioned_people(deps1)
    await wait_deferred()
    deps2 = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
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
        eager_filter_repositories=False,
    )
    people2 = deployment_facts_extract_mentioned_people(deps2)
    assert len(deps1) == len(deps2) == 2 * 9
    assert_index_equal(deps1.index, deps2.index)
    lensum = 0
    for i in range(18):
        assert (rel1 := deps1["releases"][i]).columns == (rel2 := deps2["releases"][i]).columns, i
        assert len(rel1) == len(rel2)
        for il1, il2 in zip(rel1.index.levels(), rel2.index.levels()):
            assert_array_equal(il1, il2)
        lensum += len(rel1)
    assert lensum == 68 * 2
    for topic in ("components", "labels"):
        for pdf, sdf in zip(deps1[topic], deps2[topic]):
            assert_frame_equal(pdf, sdf)
    for df in (deps1, deps2):
        for col in ("releases", "components", "labels"):
            del df[col]
    assert_frame_equal(deps1, deps2)
    assert (people1 == people2).all()


@with_defer
async def test_mine_deployments_reversed(
    sample_deployments,
    release_match_setting_tag,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
):
    time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps1 = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
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
        eager_filter_repositories=False,
    )
    deployment_facts_extract_mentioned_people(deps1)
    await wait_deferred()

    name = "%s_%d_%02d_%02d" % ("production", 2019, 12, 1)
    await rdb.execute(
        insert(DeploymentNotification).values(
            account_id=1,
            name=name,
            conclusion="SUCCESS",
            environment="production",
            started_at=datetime(2019, 12, 1, tzinfo=timezone.utc),
            finished_at=datetime(2019, 12, 1, 0, 10, tzinfo=timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    )
    await rdb.execute(
        insert(DeployedComponent).values(
            account_id=1,
            deployment_name=name,
            repository_node_id=40550,
            reference="v4.13.0",
            resolved_commit_node_id=2756276,
            created_at=datetime.now(timezone.utc),
        ),
    )

    deps2 = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
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
        eager_filter_repositories=False,
    )
    deployment_facts_extract_mentioned_people(deps2)
    assert len(deps2) == len(deps1) + 1
    assert set(deps1.index.values) == set(deps2.index.values) - {"production_2019_12_01"}
    i = np.flatnonzero(deps2.index.values == "production_2019_12_01")[0]
    assert deps2.iloc[i][DeploymentFacts.f.commits_overall][0] == 0
    assert deps2.iloc[i][DeploymentFacts.f.commits_prs][0] == 0
    assert len(deps2.iloc[i][DeploymentFacts.f.prs]) == 0


@with_defer
async def test_mine_deployments_empty(
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
):
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production", "staging"],
        [DeploymentConclusion.SUCCESS],
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
        cache,
    )
    assert len(deps) == 0
    await wait_deferred()
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production", "staging"],
        [DeploymentConclusion.SUCCESS],
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
        None,
        None,
        None,
        cache,
    )
    assert len(deps) == 0


@pytest.mark.parametrize("with_premining", [True, False])
@with_defer
async def test_mine_deployments_event_releases(
    sample_deployments,
    release_match_setting_event,
    branches,
    default_branches,
    prefixer,
    mdb,
    pdb,
    rdb,
    cache,
    with_premining,
):
    with_premining = True
    await rdb.execute(
        insert(ReleaseNotification).values(
            ReleaseNotification(
                account_id=1,
                repository_node_id=40550,
                commit_hash_prefix="1edb992",
                name="Pushed!",
                author_node_id=40020,
                url="www",
                published_at=datetime(2019, 9, 1, tzinfo=timezone.utc),
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    await rdb.execute(
        insert(ReleaseNotification).values(
            ReleaseNotification(
                account_id=1,
                repository_node_id=40550,
                commit_hash_prefix="0023a4a5d4aba74240e5bbc403e56af349edf66c",
                name="for test",
                author_node_id=40020,
                url="www",
                published_at=datetime(2019, 10, 1, tzinfo=timezone.utc),
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    if with_premining:
        await mine_releases(
            ["src-d/go-git"],
            {},
            branches,
            default_branches,
            time_from,
            time_to,
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_match_setting_event,
            LogicalRepositorySettings.empty(),
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            rdb,
            None,
            with_avatars=False,
            with_extended_pr_details=False,
            with_deployments=False,
        )
        await wait_deferred()
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production", "staging"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_event,
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
        cache,
    )
    deployment_facts_extract_mentioned_people(deps)
    for depname in ("production_2019_11_01", "staging_2019_11_01"):
        df = deps.take(deps.index.get_level_values(0) == depname)["releases"][0]
        assert len(df) == 1
        df = df.iloc[0]
        assert df[Release.name.name] == "Pushed!"
        assert df[Release.sha.name] == b"1edb992dbc419a0767b1cf3a524b0d35529799f5"
        assert len(df[ReleaseFacts.f.commit_authors]) == 113
        assert len(set(df[ReleaseFacts.f.commit_authors])) == 113
        assert len(df[ReleaseFacts.f.prs_node_id]) == 509
        assert len(set(df[ReleaseFacts.f.prs_node_id])) == 509


@pytest.mark.parametrize("old_notifications_mode", ["1", "2", "1-1"])
@with_defer
async def test_invalidate_newer_deploys_smoke(
    branches,
    default_branches,
    release_match_setting_tag_or_branch,
    prefixer,
    mdb,
    pdb,
    rdb,
    old_notifications_mode,
) -> None:
    await rdb.execute(sa.delete(DeploymentNotification))
    await rdb.execute(sa.delete(DeployedComponent))
    await rdb.execute(sa.delete(DeployedLabel))

    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 2, 1, tzinfo=timezone.utc)
    mine_releases_ = partial(
        mine_releases,
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_avatars=False,
        with_extended_pr_details=False,
        with_deployments=False,
    )

    mine_deployments_ = partial(
        mine_deployments,
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production"],
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
    await models_insert(
        rdb,
        DeploymentNotificationFactory(
            name="deploy_old_1", started_at=dt(2019, 1, 1), finished_at=dt(2019, 1, 1, 0, 10),
        ),
        DeployedComponentFactory(
            deployment_name="deploy_old_1",
            repository_node_id=40550,
            resolved_commit_node_id=2756591,
        ),
    )
    match old_notifications_mode:
        case "2":
            await models_insert(
                rdb,
                DeploymentNotificationFactory(
                    name="deploy_old_2",
                    started_at=dt(2020, 1, 1),
                    finished_at=dt(2020, 1, 1, 0, 10),
                ),
                DeployedComponentFactory(
                    deployment_name="deploy_old_2",
                    repository_node_id=40550,
                    resolved_commit_node_id=2755244,
                ),
            )
        case "1-1":
            await models_insert(
                rdb,
                DeploymentNotificationFactory(
                    name="deploy_old_2",
                    started_at=dt(2017, 1, 1),
                    finished_at=dt(2017, 1, 1, 0, 10),
                ),
                DeployedComponentFactory(
                    deployment_name="deploy_old_2",
                    repository_node_id=40550,
                    resolved_commit_node_id=2755513,
                ),
            )

    await mine_releases_()
    await wait_deferred()
    await mine_deployments_()
    await wait_deferred()

    # the new notification is about an older commit
    await models_insert(
        rdb,
        DeploymentNotificationFactory(
            name="deploy_new", started_at=dt(2018, 8, 1), finished_at=dt(2018, 8, 1, 0, 10),
        ),
        DeployedComponentFactory(
            deployment_name="deploy_new",
            repository_node_id=40550,
            resolved_commit_node_id=2755028,
        ),
    )
    deps = await mine_deployments_()
    deploy_new_prs_post = deps[DeploymentFacts.f.prs][
        deps[DeploymentFacts.f.name] == "deploy_new"
    ][0]
    deploy_old1_prs_post = deps[DeploymentFacts.f.prs][
        deps[DeploymentFacts.f.name] == "deploy_old_1"
    ][0]

    # the two deployments must not share any PR
    assert len(deploy_new_prs_post)
    assert len(deploy_old1_prs_post)
    common = set(deploy_new_prs_post) & set(deploy_old1_prs_post)
    assert not common

    if old_notifications_mode != "1":
        deploy_old2_prs_post = deps[DeploymentFacts.f.prs][
            deps[DeploymentFacts.f.name] == "deploy_old_2"
        ][0]
        assert len(deploy_old2_prs_post)
        common = set(deploy_new_prs_post) & set(deploy_old2_prs_post)
        assert not common
        common = set(deploy_old1_prs_post) & set(deploy_old2_prs_post)
        assert not common


@with_defer
async def test_invalidate_newer_deploys_in_pdb_different_envs(
    branches,
    default_branches,
    release_match_setting_tag_or_branch,
    prefixer,
    mdb,
    pdb,
    rdb,
):
    await rdb.execute(sa.delete(DeploymentNotification))
    await rdb.execute(sa.delete(DeployedComponent))
    await rdb.execute(sa.delete(DeployedLabel))

    time_from = datetime(2017, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)

    await models_insert(
        rdb,
        DeploymentNotificationFactory(
            name="deploy-prod",
            environment="production",
            started_at=dt(2019, 11, 1),
            finished_at=dt(2019, 11, 1, 0, 10),
        ),
        DeployedComponentFactory(
            deployment_name="deploy-prod",
            repository_node_id=40550,
            resolved_commit_node_id=2755244,
        ),
        DeploymentNotificationFactory(
            name="deploy-stage",
            environment="stage",
            started_at=dt(2018, 8, 1),
            finished_at=dt(2018, 8, 1, 0, 10),
        ),
        DeployedComponentFactory(
            deployment_name="deploy-stage",
            repository_node_id=40550,
            resolved_commit_node_id=2755028,
        ),
    )

    await mine_releases(
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_avatars=False,
        with_extended_pr_details=False,
        with_deployments=False,
    )

    await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production"],
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
    await wait_deferred()

    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production", "stage"],
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
    assert sorted(deps.index.values) == ["deploy-prod", "deploy-stage"]


async def _validate_deployed_prs(pdb: morcilla.Database) -> None:
    rows = await pdb.fetch_all(
        select(GitHubPullRequestDeployment.pull_request_id)
        .where(
            GitHubPullRequestDeployment.deployment_name.like("production_%"),
            GitHubPullRequestDeployment.deployment_name.notin_(
                ["production_2018_01_10", "production_2018_01_12"],
            ),
        )
        .group_by(GitHubPullRequestDeployment.pull_request_id)
        .having(func.count(GitHubPullRequestDeployment.deployment_name) > 1),
    )
    assert len(rows) == 0, rows


def _validate_deployments(deps, count, with_2016, eager):
    assert len(deps) == count * 2
    for env in ("staging", "production"):
        assert (
            deps["conclusion"][deps.index.get_level_values(0) == f"{env}_2018_01_11"]
            == DeploymentNotification.CONCLUSION_SUCCESS
        )
        if not eager:
            assert (
                deps["conclusion"][deps.index.get_level_values(0) == f"{env}_2018_01_12"]
                == DeploymentNotification.CONCLUSION_FAILURE
            )
    assert (deps["environment"] == "production").sum() == count
    assert (deps["environment"] == "staging").sum() == count
    components = deps["components"]
    for c in components:
        assert len(c) == 1
        assert c.iloc[0]["repository_node_id"] == 40550
        assert c.iloc[0]["resolved_commit_node_id"] > 0
    assert components[deps.index.values == "production_2018_01_11"][0]["reference"] == "4.0.0"
    assert components[deps.index.values == "staging_2018_01_11"][0]["reference"] == "4.0.0"
    commits_overall = deps["commits_overall"]
    if with_2016:
        assert commits_overall[deps.index.values == "production_2016_07_06"] == [168]
        assert commits_overall[deps.index.values == "production_2016_12_01"] == [14]
    assert commits_overall[deps.index.values == "production_2018_01_10"] == [832]
    assert commits_overall[deps.index.values == "production_2018_01_11"] == [832]
    if not eager:
        assert commits_overall[deps.index.values == "production_2018_01_12"] == [0]
    assert commits_overall[deps.index.values == "production_2018_08_01"] == [122]
    assert commits_overall[deps.index.values == "production_2018_12_01"] == [198]
    if not eager:
        assert commits_overall[deps.index.values == "production_2018_12_02"] == [0]
    assert commits_overall[deps.index.values == "production_2019_11_01"] == [176]
    pdeps = deps.take(deps["environment"] == "production").copy()
    releases = pdeps["releases"]
    if with_2016:
        assert set(releases[pdeps.index.values == "production_2016_07_06"][0]["tag"]) == {
            "v2.2.0",
            "v3.1.0",
            "v3.0.3",
            "v3.0.1",
            "v1.0.0",
            "v3.0.2",
            "v3.0.4",
            "v2.1.1",
            "v2.0.0",
            "v3.0.0",
            "v2.1.2",
            "v2.1.3",
            "v2.1.0",
        }
        assert set(releases[pdeps.index.values == "production_2016_12_01"][0]["tag"]) == {
            "v3.2.0",
            "v3.1.1",
        }
    assert set(releases[pdeps.index.values == "production_2018_01_10"][0]["tag"]) == {
        "v4.0.0-rc10",
        "v4.0.0-rc1",
        "v4.0.0-rc6",
        "v4.0.0-rc8",
        "v4.0.0-rc7",
        "v4.0.0-rc9",
        "v4.0.0",
        "v4.0.0-rc13",
        "v4.0.0-rc2",
        "v4.0.0-rc12",
        "v4.0.0-rc14",
        "v4.0.0-rc15",
        "v4.0.0-rc3",
        "v4.0.0-rc5",
        "v4.0.0-rc4",
        "v4.0.0-rc11",
    }
    assert set(releases[pdeps.index.values == "production_2018_01_11"][0]["tag"]) == {
        "v4.0.0-rc10",
        "v4.0.0-rc1",
        "v4.0.0-rc6",
        "v4.0.0-rc8",
        "v4.0.0-rc7",
        "v4.0.0-rc9",
        "v4.0.0",
        "v4.0.0-rc13",
        "v4.0.0-rc2",
        "v4.0.0-rc12",
        "v4.0.0-rc14",
        "v4.0.0-rc15",
        "v4.0.0-rc3",
        "v4.0.0-rc5",
        "v4.0.0-rc4",
        "v4.0.0-rc11",
    }
    if not eager:
        assert len(releases[pdeps.index.values == "production_2018_01_12"][0]) == 0
    assert set(releases[pdeps.index.values == "production_2018_08_01"][0]["tag"]) == {
        "v4.3.1",
        "v4.5.0",
        "v4.4.0",
        "v4.3.0",
        "v4.2.0",
        "v4.4.1",
        "v4.2.1",
        "v4.1.0",
        "v4.1.1",
    }
    assert set(releases[pdeps.index.values == "production_2018_12_01"][0]["tag"]) == {
        "v4.7.1",
        "v4.6.0",
        "v4.8.0",
        "v4.8.1",
        "v4.7.0",
    }
    if not eager:
        assert len(releases[pdeps.index.values == "production_2018_12_02"][0]) == 0
    assert set(releases[pdeps.index.values == "production_2019_11_01"][0]["tag"]) == {
        "v4.13.0",
        "v4.12.0",
        "v4.13.1",
        "v4.9.0",
        "v4.9.1",
        "v4.11.0",
        "v4.10.0",
    }
    pdeps.sort_index(inplace=True)
    pdeps.reset_index(inplace=True, drop=True)
    del pdeps["environment"]
    sdeps = deps.take(deps["environment"] == "staging").copy()
    sdeps.sort_index(inplace=True)
    sdeps.reset_index(inplace=True, drop=True)
    del sdeps["environment"]
    for topic in ("releases", "components", "labels"):
        for pdf, sdf in zip(pdeps[topic], sdeps[topic]):
            assert_frame_equal(pdf, sdf)
    for df in (pdeps, sdeps):
        for col in ("releases", "components", "labels"):
            del df[col]
    assert_frame_equal(pdeps, sdeps)


class TestHideOutlierFirstDeployments:
    @with_defer
    async def test_base(
        self,
        sample_deployments,
        branches,
        prefixer,
        mdb,
        pdb,
        rdb,
        sdb,
        default_branches,
        release_match_setting_tag,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        assert (await count(pdb, GitHubDeploymentFacts)) == 0
        assert (await count(pdb, GitHubPullRequestDeployment)) == 0

        deps = await mine_deployments(
            **self._mine_common_kwargs(default_branches, release_match_setting_tag),
            branches=branches,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
        )
        # wait deferred tasks writing to pdb to complete
        await wait_deferred()

        expected_outlier_deployments = ["staging_2016_07_06", "production_2016_07_06"]

        pr_where = GitHubPullRequestDeployment.deployment_name.in_(expected_outlier_deployments)
        pre_hiding_pr_count = await count(pdb, GitHubPullRequestDeployment)
        to_be_removed_pr_count = await count(pdb, GitHubPullRequestDeployment, pr_where)

        commit_where = GitHubCommitDeployment.deployment_name.in_(expected_outlier_deployments)
        pre_hiding_commit_count = await count(pdb, GitHubCommitDeployment)
        to_be_removed_commit_count = await count(pdb, GitHubCommitDeployment, commit_where)

        rel_where = GitHubReleaseDeployment.deployment_name.in_(expected_outlier_deployments)
        pre_hiding_release_count = await count(pdb, GitHubReleaseDeployment)
        to_be_removed_release_count = await count(pdb, GitHubReleaseDeployment, rel_where)

        await hide_outlier_first_deployments(deps, 1, meta_ids, mdb, pdb, 1.1)

        post_hiding_pr_count = await count(pdb, GitHubPullRequestDeployment)
        post_hiding_commit_count = await count(pdb, GitHubCommitDeployment)
        post_hiding_release_count = await count(pdb, GitHubReleaseDeployment)

        assert post_hiding_pr_count == pre_hiding_pr_count - to_be_removed_pr_count
        assert post_hiding_commit_count == pre_hiding_commit_count - to_be_removed_commit_count
        assert post_hiding_release_count == pre_hiding_release_count - to_be_removed_release_count

        assert await count(pdb, GitHubPullRequestDeployment, pr_where) == 0

        depl_where = GitHubDeploymentFacts.deployment_name.in_(expected_outlier_deployments)
        depl_rows = await pdb.fetch_all(sa.select(GitHubDeploymentFacts).where(depl_where))
        for depl_row in depl_rows:
            facts = DeploymentFacts(
                depl_row[GitHubDeploymentFacts.data.name],
                name=depl_row[GitHubDeploymentFacts.deployment_name.name],
            )
            assert not len(facts.prs)

    @with_defer
    async def test_no_deployment_facts(
        self,
        branches,
        prefixer,
        mdb,
        pdb,
        rdb,
        sdb,
        default_branches,
        release_match_setting_tag,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        # delete notifications so that mine_deployments will find nothing
        await rdb.execute(sa.delete(DeploymentNotification))

        deps = await mine_deployments(
            **self._mine_common_kwargs(default_branches, release_match_setting_tag),
            branches=branches,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
        )

        assert deps.empty
        # wait deferred tasks writing to pdb to complete
        await wait_deferred()

        assert (await count(pdb, GitHubPullRequestDeployment)) == 0
        assert (await count(pdb, GitHubCommitDeployment)) == 0
        assert (await count(pdb, GitHubReleaseDeployment)) == 0

        await hide_outlier_first_deployments(deps, 1, meta_ids, mdb, pdb, 1.1)

    @with_defer
    async def test_multiple_deployments_same_time(
        self,
        branches,
        prefixer,
        mdb,
        pdb,
        rdb,
        sdb,
        default_branches,
        release_match_setting_tag,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        await rdb.execute(sa.delete(DeploymentNotification))
        started_at = dt(2019, 1, 1)
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="deploy0", started_at=started_at),
            DeploymentNotificationFactory(name="deploy1", started_at=started_at),
            DeployedComponentFactory(deployment_name="deploy0", repository_node_id=40550),
            DeployedComponentFactory(deployment_name="deploy1", repository_node_id=40550),
        )

        deps = await mine_deployments(
            **self._mine_common_kwargs(default_branches, release_match_setting_tag),
            branches=branches,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            eager_filter_repositories=False,
        )
        await wait_deferred()

        await hide_outlier_first_deployments(deps, 1, meta_ids, mdb, pdb, 1.1)

        await assert_missing_row(pdb, GitHubPullRequestDeployment, deployment_name="deploy0")

    @with_defer
    async def test_logical_deployments(
        self,
        branches,
        prefixer,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag_logical,
        default_branches,
    ) -> None:
        logical_settings = LogicalRepositorySettings(
            {"src-d/go-git/alpha": {"title": "alpha-.*"}},
            {"src-d/go-git/alpha": {"title": "alpha-.*"}},
        )

        await rdb.execute(sa.delete(DeploymentNotification))
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="alpha-0", started_at=dt(2019, 1, 1)),
            DeployedComponentFactory(deployment_name="alpha-0", repository_node_id=40550),
            DeploymentNotificationFactory(name="alpha-1", started_at=dt(2019, 1, 2)),
            DeployedComponentFactory(deployment_name="alpha-1", repository_node_id=40550),
        )
        deps = await mine_deployments(
            **self._mine_common_kwargs(
                default_branches,
                release_match_setting_tag_logical,
                logical_settings=logical_settings,
                repositories=["src-d/go-git/alpha"],
            ),
            branches=branches,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
        )
        await wait_deferred()

        await hide_outlier_first_deployments(deps, 1, (6366825,), mdb, pdb, 1.1)

        await assert_missing_row(pdb, GitHubPullRequestDeployment, deployment_name="alpha-0")

    async def test_handle_unknown_repository(self, mdb, pdb) -> None:
        success = DeploymentNotification.CONCLUSION_SUCCESS
        df = self._mk_deployments_df(
            ("deploy0", "prod", dt(2020, 1, 1), success, ["org/repo"]),
            ("deploy1", "prod", dt(2020, 1, 2), success, ["org/repo"]),
        )
        await hide_outlier_first_deployments(df, 1, (DEFAULT_MD_ACCOUNT_ID,), mdb, pdb, 1.1)

    @classmethod
    def _mk_deployments_df(cls, *rows) -> md.DataFrame:
        df_columns = [
            DeploymentNotification.name.name,
            DeploymentNotification.environment.name,
            DeploymentNotification.started_at.name,
            DeploymentNotification.conclusion.name,
            "repositories",
        ]
        arrays = [[] for _ in df_columns]
        for r in rows:
            for i, v in enumerate(r):
                if i == 2:
                    v = v.replace(tzinfo=None)
                arrays[i].append(v)
        repos = np.empty(len(rows), object)
        repos[:] = arrays[-1]
        arrays[-1] = repos
        return md.DataFrame(dict(zip(df_columns, arrays)))

    @classmethod
    def _mine_common_kwargs(
        cls,
        default_branches,
        release_settings,
        **extra: Any,
    ) -> dict:
        return {
            "repositories": ["src-d/go-git"],
            "participants": {},
            "time_from": datetime(2015, 1, 1, tzinfo=timezone.utc),
            "time_to": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "environments": ["production", "staging"],
            "conclusions": [],
            "with_labels": {},
            "without_labels": {},
            "pr_labels": LabelFilter.empty(),
            "jira": JIRAFilter.empty(),
            "release_settings": release_settings,
            "logical_settings": LogicalRepositorySettings.empty(),
            "default_branches": default_branches,
            "account": 1,
            "jira_ids": JIRAConfig(1, {}, {}),
            "meta_ids": (6366825,),
            "cache": None,
            **extra,
        }


async def test_reset_broken_deployments_dupe(precomputed_deployments, pdb, rdb):  # noqa: F811
    await gather(
        rdb.execute(
            insert(DeploymentNotification).values(
                account_id=1,
                name="xxx",
                conclusion=DeploymentNotification.CONCLUSION_SUCCESS.decode(),
                environment="production",
                started_at=datetime(2023, 1, 12, tzinfo=timezone.utc),
                finished_at=datetime(2023, 1, 12, 0, 10, tzinfo=timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        ),
        rdb.execute(
            insert(DeploymentNotification).values(
                account_id=1,
                name="yyy",
                conclusion=DeploymentNotification.CONCLUSION_FAILURE.decode(),
                environment="production",
                started_at=datetime(2023, 1, 11, tzinfo=timezone.utc),
                finished_at=datetime(2023, 1, 11, 0, 10, tzinfo=timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        ),
        pdb.execute(
            insert(GitHubCommitDeployment).values(
                {
                    GitHubCommitDeployment.acc_id: 1,
                    GitHubCommitDeployment.repository_full_name: "src-d/go-git",
                    GitHubCommitDeployment.commit_id: 2755079,
                    GitHubCommitDeployment.deployment_name: "xxx",
                },
            ),
        ),
        pdb.execute(
            insert(GitHubCommitDeployment).values(
                {
                    GitHubCommitDeployment.acc_id: 1,
                    GitHubCommitDeployment.repository_full_name: "src-d/go-git",
                    GitHubCommitDeployment.commit_id: 2755079,
                    GitHubCommitDeployment.deployment_name: "yyy",
                },
            ),
        ),
    )
    assert await reset_broken_deployments(1, pdb, rdb) == 2
    for name in ("xxx", "Dummy deployment"):
        await assert_missing_row(pdb, GitHubCommitDeployment, deployment_name=name)
    await assert_existing_row(pdb, GitHubCommitDeployment, deployment_name="yyy")


async def test_reset_broken_deployments_wtf(pdb, rdb):
    await pdb.execute(
        insert(GitHubCommitDeployment).values(
            {
                GitHubCommitDeployment.acc_id: 1,
                GitHubCommitDeployment.repository_full_name: "src-d/go-git",
                GitHubCommitDeployment.commit_id: 2755079,
                GitHubCommitDeployment.deployment_name: "xxx",
            },
        ),
    )
    assert await reset_broken_deployments(1, pdb, rdb) == 1


async def test_extract_pr_commits_smoke(dag):
    dag = dag["src-d/go-git"][1]
    pr_commits = extract_pr_commits(
        *dag,
        np.array(
            [
                b"e20d3347d26f0b7193502e2ad7386d7c504b0cde",
                b"6dda959c4bda3a422a9a1c6425f92efa914c4d82",
                b"0000000000000000000000000000000000000000",
            ],
            dtype="S40",
        ),
    )
    assert pr_commits[0].tolist() == [
        b"e20d3347d26f0b7193502e2ad7386d7c504b0cde",
        b"3452c3bde5c0bddfd52bb827d07d4a1e1ed3fb09",
    ]
    assert set(pr_commits[1]) == {
        b"6dda959c4bda3a422a9a1c6425f92efa914c4d82",
        b"9c80677ec0d1778e6d304b235a22f4e636322e74",
        b"129ff16f8686bb74a206cf58000de1d9640e370a",
        b"c8ca7e3d031214b6c0478d62119dfb8a9af1631d",
    }
    assert pr_commits[2].tolist() == []
    assert extract_pr_commits(*dag, np.array([], dtype="S40")).tolist() == []


async def test_extract_pr_commits_fixture():
    with Path(__file__).with_name("extract_pr_commits.pickle").open("rb") as fin:
        args = pickle.load(fin)
    extract_pr_commits(*args)


async def test_extract_independent_ownership_no_stops(dag):
    dag = dag["src-d/go-git"][1]
    stops = np.empty(4, dtype=object)
    stops.fill([])
    ownership = extract_independent_ownership(
        *dag,
        np.array(
            [
                b"b65d94e70ea1d013f43234522fa092168e4f1041",
                b"3713157d189a109bdccdb055200defb17297b6de",
                b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff",
                b"0000000000000000000000000000000000000000",
            ],
            dtype="S40",
        ),
        stops,
    )
    assert len(ownership[0]) == 443
    assert len(ownership[1]) == 603
    assert len(ownership[2]) == 3
    assert len(ownership[3]) == 0


async def test_extract_independent_ownership_smoke(dag):
    dag = dag["src-d/go-git"][1]
    ownership = extract_independent_ownership(
        *dag,
        np.array(
            [
                b"b65d94e70ea1d013f43234522fa092168e4f1041",
                b"3713157d189a109bdccdb055200defb17297b6de",
                b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff",
                b"0000000000000000000000000000000000000000",
            ],
            dtype="S40",
        ),
        np.array(
            [
                [b"431af32445562b389397f3ee7af90bf61455fff1"],
                [b"e80cdbabb92a1ec35ffad536f52d3ff04b548fd1"],
                [b"0000000000000000000000000000000000000000"],
                [b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff"],
            ],
            dtype="S40",
        ),
    )
    assert ownership[3].tolist() == []
    assert set(ownership[2]) == {
        b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff",
        b"5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf",
        b"5d7303c49ac984a9fec60523f2d5297682e16646",
    }
    assert set(ownership[1]) == {
        b"3713157d189a109bdccdb055200defb17297b6de",
        b"b8b61e74469e0d2662e7d690eee14893f91fe259",
        b"40fa5882a2c73f8c075403b7ec85870f04deda07",
        b"ff18ce3751ad80cfd0297845872ba1d796c36ca5",
        b"5592dabdf9eed67c92b0e411ad375ae763119fd2",
        b"2e092f909f643ef455d84dfa59282f0f0adf3c7a",
        b"9a807f4f29c24bf0dc0b44d7cdfc2233cfd128d3",
    }
    assert set(ownership[0]) == {
        b"b65d94e70ea1d013f43234522fa092168e4f1041",
        b"84b6bd8c22c8683479881a67db03dfdeeeb299ce",
        b"f3554ac62e29fd3e06c6e1fdeda914ac19cb68fa",
        b"77a8f2bfbd2b19d6e19edeb7a9276bc9fe576c00",
        b"62666856d9f4b3150671eb1f215a7072c02c0bc6",
        b"1f39465975d56bbb02f5cdfb1e3e77f41c613f1d",
        b"2b1efd219e1f20d9a0bc380a26074c9d8de2ae1f",
        b"70eff0d7bd1f69856f8028c2487576085a54a42c",
        b"c340fb9a0f1f7c025da5ffa2d1a7389a4eabaae2",
    }
    assert (
        extract_independent_ownership(
            *dag, np.array([], dtype="S40"), np.array([], dtype="S40"),
        ).tolist()
        == []
    )


class TestInvalidatePrecomputedOnLabelsChange:
    async def test_no_invalidation(
        self,
        sdb: Database,
        mdb_rw: Database,
        rdb: Database,
        pdb: Database,
    ) -> None:
        await models_insert(pdb, GitHubDeploymentFactsFactory(deployment_name="d"))
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d"),
            DeployedComponentFactory(deployment_name="d", repository_node_id=99),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            prefixer = await Prefixer.load((DEFAULT_MD_ACCOUNT_ID,), mdb_rw, None)
            settings = Settings.from_account(1, prefixer, sdb, mdb_rw, None, None)
            logical_settings = await settings.list_logical_repositories()

            await invalidate_precomputed_on_labels_change(
                "d", ["l0"], 1, prefixer, logical_settings, rdb, pdb,
            )

        await assert_existing_row(pdb, GitHubDeploymentFacts, deployment_name="d")

    async def test_invalidation(
        self,
        sdb: Database,
        mdb_rw: Database,
        rdb: Database,
        pdb: Database,
    ) -> None:
        await models_insert(
            pdb,
            GitHubDeploymentFactsFactory(deployment_name="d"),
            GitHubDeploymentFactsFactory(deployment_name="d2"),
            GitHubDeploymentFactsFactory(deployment_name="d3"),
        )
        await models_insert(
            rdb,
            DeploymentNotificationFactory(name="d", finished_at=dt(2021, 1, 2)),
            DeployedComponentFactory(deployment_name="d", repository_node_id=99),
            DeployedLabelFactory(deployment_name="d", key="k"),
            DeploymentNotificationFactory(name="d2", finished_at=dt(2021, 1, 3)),
            DeployedComponentFactory(deployment_name="d2", repository_node_id=99),
            DeploymentNotificationFactory(name="d3", finished_at=dt(2021, 1, 1)),
            DeployedComponentFactory(deployment_name="d3", repository_node_id=99),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            await insert_logical_repo(99, "l", sdb, deployments={"labels": {"lab0": "v"}})
            prefixer = await Prefixer.load((DEFAULT_MD_ACCOUNT_ID,), mdb_rw, None)
            settings = Settings.from_account(1, prefixer, sdb, mdb_rw, None, None)
            logical_settings = await settings.list_logical_repositories()

            await invalidate_precomputed_on_labels_change(
                "d", ["lab0"], 1, prefixer, logical_settings, rdb, pdb,
            )

        await assert_missing_row(pdb, GitHubDeploymentFacts, deployment_name="d")
        # d2 invalidated, finished after d
        await assert_missing_row(pdb, GitHubDeploymentFacts, deployment_name="d2")
        # d not invalidated, finished before d
        await assert_existing_row(pdb, GitHubDeploymentFacts, deployment_name="d3")
