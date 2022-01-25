from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle

import morcilla
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import and_, delete, func, insert, select

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.dag_accelerated import extract_independent_ownership, \
    extract_pr_commits
from athenian.api.controllers.miners.github.deployment import mine_deployments
from athenian.api.controllers.miners.github.release_mine import mine_releases, \
    mine_releases_by_name
from athenian.api.controllers.miners.types import DeployedComponent as DeployedComponentStruct, \
    Deployment, DeploymentConclusion, DeploymentFacts
from athenian.api.controllers.settings import LogicalRepositorySettings
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import Release
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel, \
    DeploymentNotification, ReleaseNotification
from athenian.api.models.precomputed.models import GitHubCommitDeployment, \
    GitHubPullRequestDeployment


async def test_extract_pr_commits_smoke(dag):
    dag = dag["src-d/go-git"]
    pr_commits = extract_pr_commits(*dag, np.array([
        b"e20d3347d26f0b7193502e2ad7386d7c504b0cde",
        b"6dda959c4bda3a422a9a1c6425f92efa914c4d82",
        b"0000000000000000000000000000000000000000",
    ], dtype="S40"))
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
    dag = dag["src-d/go-git"]
    stops = np.empty(4, dtype=object)
    stops.fill([])
    ownership = extract_independent_ownership(*dag, np.array([
        b"b65d94e70ea1d013f43234522fa092168e4f1041",
        b"3713157d189a109bdccdb055200defb17297b6de",
        b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff",
        b"0000000000000000000000000000000000000000",
    ], dtype="S40"), stops)
    assert len(ownership[0]) == 443
    assert len(ownership[1]) == 603
    assert len(ownership[2]) == 3
    assert len(ownership[3]) == 0


async def test_extract_independent_ownership_smoke(dag):
    dag = dag["src-d/go-git"]
    ownership = extract_independent_ownership(*dag, np.array([
        b"b65d94e70ea1d013f43234522fa092168e4f1041",
        b"3713157d189a109bdccdb055200defb17297b6de",
        b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff",
        b"0000000000000000000000000000000000000000",
    ], dtype="S40"), np.array([
        [b"431af32445562b389397f3ee7af90bf61455fff1"],
        [b"e80cdbabb92a1ec35ffad536f52d3ff04b548fd1"],
        [b"0000000000000000000000000000000000000000"],
        [b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff"],
    ], dtype="S40"))
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
    assert extract_independent_ownership(
        *dag, np.array([], dtype="S40"), np.array([], dtype="S40"),
    ).tolist() == []


def _validate_deployments(deps, count, with_2016):
    assert len(deps) == count * 2
    for env in ("staging", "production"):
        assert deps.loc[f"{env}_2018_01_11"]["conclusion"] == "SUCCESS"
        assert deps.loc[f"{env}_2018_01_12"]["conclusion"] == "FAILURE"
    assert (deps["environment"] == "production").sum() == count
    assert (deps["environment"] == "staging").sum() == count
    components = deps["components"]
    for c in components:
        assert len(c) == 1
        assert c.iloc[0]["repository_node_id"] == 40550
        assert c.iloc[0]["resolved_commit_node_id"] > 0
    assert components["production_2018_01_11"].iloc[0]["reference"] == "4.0.0"
    assert components["staging_2018_01_11"].iloc[0]["reference"] == "4.0.0"
    commits_overall = deps["commits_overall"]
    if with_2016:
        assert commits_overall["production_2016_07_06"] == [168]
        assert commits_overall["production_2016_12_01"] == [14]
    assert commits_overall["production_2018_01_10"] == [832]
    assert commits_overall["production_2018_01_11"] == [832]
    assert commits_overall["production_2018_01_12"] == [0]
    assert commits_overall["production_2018_08_01"] == [122]
    assert commits_overall["production_2018_12_01"] == [198]
    assert commits_overall["production_2018_12_02"] == [0]
    assert commits_overall["production_2019_11_01"] == [176]
    pdeps = deps[deps["environment"] == "production"].copy()
    releases = pdeps["releases"]
    if with_2016:
        assert set(releases["production_2016_07_06"]["tag"]) == {
            "v2.2.0", "v3.1.0", "v3.0.3", "v3.0.1", "v1.0.0", "v3.0.2", "v3.0.4", "v2.1.1",
            "v2.0.0", "v3.0.0", "v2.1.2", "v2.1.3", "v2.1.0",
        }
        assert set(releases["production_2016_12_01"]["tag"]) == {"v3.2.0", "v3.1.1"}
    assert set(releases["production_2018_01_10"]["tag"]) == {
        "v4.0.0-rc10", "v4.0.0-rc1", "v4.0.0-rc6", "v4.0.0-rc8", "v4.0.0-rc7", "v4.0.0-rc9",
        "v4.0.0", "v4.0.0-rc13", "v4.0.0-rc2", "v4.0.0-rc12", "v4.0.0-rc14", "v4.0.0-rc15",
        "v4.0.0-rc3", "v4.0.0-rc5", "v4.0.0-rc4", "v4.0.0-rc11",
    }
    assert set(releases["production_2018_01_11"]["tag"]) == {
        "v4.0.0-rc10", "v4.0.0-rc1", "v4.0.0-rc6", "v4.0.0-rc8", "v4.0.0-rc7", "v4.0.0-rc9",
        "v4.0.0", "v4.0.0-rc13", "v4.0.0-rc2", "v4.0.0-rc12", "v4.0.0-rc14", "v4.0.0-rc15",
        "v4.0.0-rc3", "v4.0.0-rc5", "v4.0.0-rc4", "v4.0.0-rc11",
    }
    assert releases["production_2018_01_12"].empty
    assert set(releases["production_2018_08_01"]["tag"]) == {
        "v4.3.1", "v4.5.0", "v4.4.0", "v4.3.0", "v4.2.0", "v4.4.1", "v4.2.1", "v4.1.0", "v4.1.1",
    }
    assert set(releases["production_2018_12_01"]["tag"]) == {
        "v4.7.1", "v4.6.0", "v4.8.0", "v4.8.1", "v4.7.0",
    }
    assert releases["production_2018_12_02"].empty
    assert set(releases["production_2019_11_01"]["tag"]) == {
        "v4.13.0", "v4.12.0", "v4.13.1", "v4.9.0", "v4.9.1", "v4.11.0", "v4.10.0",
    }
    del pdeps["environment"]
    pdeps.sort_index(inplace=True)
    pdeps.reset_index(drop=True, inplace=True)
    sdeps = deps[deps["environment"] == "staging"].copy()
    del sdeps["environment"]
    sdeps.sort_index(inplace=True)
    sdeps.reset_index(drop=True, inplace=True)
    for pdf, sdf in zip(pdeps["releases"].values, sdeps["releases"].values):
        assert_frame_equal(pdf, sdf)
    del pdeps["releases"]
    del sdeps["releases"]
    assert_frame_equal(pdeps, sdeps)


@with_defer
async def test_mine_deployments_from_scratch(
        sample_deployments, release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git"], {}, branches, default_branches, time_from, time_to,
        LabelFilter.empty(), JIRAFilter.empty(), release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None, with_avatars=False,
        with_pr_titles=False, with_deployments=False,
    )
    await wait_deferred()
    deps, people = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    _validate_deployments(deps, 9, True)
    await wait_deferred()
    commits = await pdb.fetch_all(select([GitHubCommitDeployment]))
    assert len(commits) == 4684


@with_defer
async def test_mine_deployments_middle(
        sample_deployments, release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    time_from = datetime(2017, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git"], {}, branches, default_branches,
        datetime(2016, 1, 1, tzinfo=timezone.utc), time_to,
        LabelFilter.empty(), JIRAFilter.empty(), release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None, with_avatars=False,
        with_pr_titles=False, with_deployments=False,
    )
    await wait_deferred()
    deps, people = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    _validate_deployments(deps, 7, False)


async def validate_deployed_prs(pdb: morcilla.Database) -> None:
    rows = await pdb.fetch_all(
        select([GitHubPullRequestDeployment.pull_request_id])
        .where(and_(GitHubPullRequestDeployment.deployment_name.like("production_%"),
                    GitHubPullRequestDeployment.deployment_name.notin_([
                        "production_2018_01_10", "production_2018_01_12",
                    ])))
        .group_by(GitHubPullRequestDeployment.pull_request_id)
        .having(func.count(GitHubPullRequestDeployment.deployment_name) > 1),
    )
    assert len(rows) == 0, rows


@with_defer
async def test_mine_deployments_append(
        sample_deployments, release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 11, 2, tzinfo=timezone.utc)
    await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    name = "%s_%d_%02d_%02d" % ("production", 2019, 11, 2)
    await rdb.execute(insert(DeploymentNotification).values(dict(
        account_id=1,
        name=name,
        conclusion="SUCCESS",
        environment="production",
        started_at=datetime(2019, 11, 2, tzinfo=timezone.utc),
        finished_at=datetime(2019, 11, 2, 0, 10, tzinfo=timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )))
    await rdb.execute(insert(DeployedComponent).values(dict(
        account_id=1,
        deployment_name=name,
        repository_node_id=40550,
        reference="v4.13.1",
        resolved_commit_node_id=2755244,
        created_at=datetime.now(timezone.utc),
    )))
    deps, _ = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to + timedelta(days=1),
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert len(deps.loc[name]["prs"]) == 0
    assert len(deps.loc[name]["releases"]) == 0
    await validate_deployed_prs(pdb)


@with_defer
async def test_mine_deployments_insert_middle(
        sample_deployments, release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 12, 31, tzinfo=timezone.utc)
    await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    time_from = datetime(2015, 12, 31, tzinfo=timezone.utc)
    time_to = datetime(2019, 12, 31, tzinfo=timezone.utc)
    await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    await validate_deployed_prs(pdb)


@with_defer
async def test_mine_deployments_only_failed(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    for year, month, day, conclusion, tag, commit in (
            (2018, 1, 10, "FAILURE", "4.0.0", 2757510),
    ):
        name = "production_%d_%02d_%02d" % (year, month, day)
        await rdb.execute(insert(DeploymentNotification).values(dict(
            account_id=1,
            name=name,
            conclusion=conclusion,
            environment="production",
            started_at=datetime(year, month, day, tzinfo=timezone.utc),
            finished_at=datetime(year, month, day, 0, 10, tzinfo=timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )))
        await rdb.execute(insert(DeployedComponent).values(dict(
            account_id=1,
            deployment_name=name,
            repository_node_id=40550,
            reference=tag,
            resolved_commit_node_id=commit,
            created_at=datetime.now(timezone.utc),
        )))
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 11, 2, tzinfo=timezone.utc)
    deps, _ = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert len(deps) == 1
    assert len(deps.iloc[0]["prs"]) == 246
    rows = await pdb.fetch_all(select([GitHubPullRequestDeployment]))
    assert len(rows) == 246


@with_defer
async def test_mine_deployments_logical(
        sample_deployments, release_match_setting_tag_logical, branches, default_branches,
        prefixer, logical_settings_full, mdb, pdb, rdb, cache):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git/alpha", "src-d/go-git/beta", "src-d/go-git"],
        {}, branches, default_branches, time_from, time_to,
        LabelFilter.empty(), JIRAFilter.empty(), release_match_setting_tag_logical,
        logical_settings_full, prefixer, 1, (6366825,), mdb, pdb, rdb, None,
        with_avatars=False, with_pr_titles=False, with_deployments=False,
    )
    await wait_deferred()
    deps, _ = await mine_deployments(
        ["src-d/go-git/alpha"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_logical, logical_settings_full,
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(deps) == 18
    physical_count = alpha_count = beta_count = beta_releases = 0
    for deployment_name, components, releases in zip(deps.index.values,
                                                     deps["components"].values,
                                                     deps["releases"].values):
        component_repos = components[DeployedComponent.repository_full_name].unique()
        release_repos = releases.index.get_level_values(1).unique() \
            if not releases.empty else np.array([], dtype=object)
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
    assert beta_count == 10
    assert physical_count == 6
    assert beta_releases == 6


@with_defer
async def test_mine_deployments_no_prs(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2016, 1, 1, tzinfo=timezone.utc)
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    await rdb.execute(insert(DeploymentNotification).values(dict(
        account_id=1,
        name="DeployWithoutPRs",
        conclusion="SUCCESS",
        environment="production",
        started_at=datetime(2015, 5, 21, tzinfo=timezone.utc),
        finished_at=datetime(2015, 5, 21, 0, 10, tzinfo=timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )))
    await rdb.execute(insert(DeployedComponent).values(dict(
        account_id=1,
        deployment_name="DeployWithoutPRs",
        repository_node_id=40550,
        reference="35b585759cbf29f8ec428ef89da20705d59f99ec",
        resolved_commit_node_id=2755715,
        created_at=datetime.now(timezone.utc),
    )))
    deps, _ = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(deps) == 1
    assert len(deps.iloc[0]["prs"]) == 0


@with_defer
async def test_mine_deployments_no_release_facts(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps, people = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(deps) == 1
    assert deps.iloc[0].name == "Dummy deployment"
    obj = deps["releases"].iloc[0].to_dict()
    for val in obj.values():
        if isinstance(val, dict):
            for r, arr in val.items():
                if isinstance(arr, np.ndarray):
                    val[r] = arr.tolist()
    assert obj == {
        "commit_authors": {
            (41475, "src-d/go-git"): [39789, 40187],
            (41474, "src-d/go-git"): [39771, 39789, 39887, 40025, 40292],
            (41473, "src-d/go-git"): [39789, 39811, 39828, 39868, 39890, 39974, 39994, 40083,
                                      40114, 40146, 40158,
                                      40187, 40410],
            (41472, "src-d/go-git"): [39789, 39849, 39974, 40020, 40138, 40167, 40243, 40298,
                                      40375],
            (41471, "src-d/go-git"): [39789, 39926, 40020, 40039, 40374],
            (41470, "src-d/go-git"): [39789, 39849],
            (41469, "src-d/go-git"): [39789, 39828, 39849, 40328],
            (41468, "src-d/go-git"): [39789, 40070],
            (41467, "src-d/go-git"): [39789, 39798, 39913, 39926, 40031, 40032, 40044, 40061,
                                      40070, 40228, 40261,
                                      40414],
            (41485, "src-d/go-git"): [39764, 39789, 39828, 39849, 39900, 39926, 39968, 40070,
                                      40095, 40181, 40186,
                                      40189, 40374, 40377, 40410],
            (41484, "src-d/go-git"): [39789, 39849, 39863, 40070, 40076, 40197, 40414],
            (41483, "src-d/go-git"): [39789, 39849, 39900, 39926, 39957, 40165, 40266, 40345,
                                      40414],
            (41482, "src-d/go-git"): [39789, 39957, 40070, 40135],
            (41481, "src-d/go-git"): [39788, 39789, 39849, 39926, 40135],
            (41480, "src-d/go-git"): [39789, 39804, 39849],
            (41479, "src-d/go-git"): [39789, 39849, 40175, 40374],
            (41478, "src-d/go-git"): [39789, 39845, 40068, 40093],
            (41477, "src-d/go-git"): [39788, 39789, 39804, 39849, 39881, 39896, 39926, 40239],
            (41476, "src-d/go-git"): [39789, 39874, 39881, 40038, 40093, 40287],
            (41517, "src-d/go-git"): [39789, 39814, 39849, 40090, 40283, 40288],
            (41519, "src-d/go-git"): [39789, 39800, 39849],
            (41518, "src-d/go-git"): [39789, 39844, 39849, 39926, 39936, 39957, 39962, 39968,
                                      39980, 39989, 40051,
                                      40052, 40059, 40109, 40170, 40175, 40197, 40230, 40239,
                                      40246, 40283, 40293,
                                      40319, 40374, 40392],
            (41514, "src-d/go-git"): [39789, 39957, 40239],
            (41513, "src-d/go-git"): [39789, 39799, 39818, 39949, 39957, 40070, 40123, 40239,
                                      40304, 40374],
            (41516, "src-d/go-git"): [39789, 39926, 40070, 40239],
            (41515, "src-d/go-git"): [39789, 39891, 39926, 40045, 40070],
            (41512, "src-d/go-git"): [39789, 39926, 40070, 40239, 40317],
            (41511, "src-d/go-git"): [39789, 39800, 39891, 39906, 39926, 40070, 40073, 40093,
                                      40175, 40284, 40385,
                                      40418],
            (41510, "src-d/go-git"): [39789],
            (41509, "src-d/go-git"): [39789, 39926, 40096, 40418],
            (41508, "src-d/go-git"): [39789, 39921, 39926, 39957, 40070, 40418],
            (41506, "src-d/go-git"): [40070, 40096],
            (41505, "src-d/go-git"): [39789, 39926, 40070, 40096, 40283, 40418],
            (41503, "src-d/go-git"): [39789, 40070, 40283, 40418],
            (41502, "src-d/go-git"): [39789, 40070, 40303],
            (41501, "src-d/go-git"): [39789, 39926, 40070, 40394, 40418],
            (41507, "src-d/go-git"): [39789, 40020, 40070, 40283, 40418],
            (41496, "src-d/go-git"): [40020, 40418],
            (41495, "src-d/go-git"): [39789, 40020, 40418],
            (41500, "src-d/go-git"): [39789],
            (41499, "src-d/go-git"): [39758, 39789],
            (41498, "src-d/go-git"): [39789],
            (41497, "src-d/go-git"): [39789, 40058],
            (41490, "src-d/go-git"): [39789, 40058],
            (41488, "src-d/go-git"): [39789, 39904, 39957, 40238, 40389, 40418],
            (41487, "src-d/go-git"): [39789, 40418],
            (41486, "src-d/go-git"): [39789, 40418],
            (41494, "src-d/go-git"): [39789, 40418],
            (41493, "src-d/go-git"): [39789, 40418],
            (41492, "src-d/go-git"): [39789, 40274, 40418],
            (41491, "src-d/go-git"): [39789, 40221, 40274, 40310, 40347, 40366]},
        "prs_node_id": {
            (41475, "src-d/go-git"): [163398],
            (41474, "src-d/go-git"): [163375, 163380, 163378, 163395, 163397, 163377, 163396],
            (41473, "src-d/go-git"): [163373, 163334, 163347, 163363, 163357, 163356, 163367,
                                      163333,
                                      163341, 163348, 163351, 163346, 163364, 163374, 163360,
                                      163372,
                                      163371, 163365, 163370, 163359, 163369, 163366, 163368,
                                      163362,
                                      163361],
            (41472, "src-d/go-git"): [163324, 163318, 163332, 163322, 163340, 163342, 163354,
                                      163325,
                                      163319, 163336, 163326, 163343, 163344, 163353, 163320],
            (41471, "src-d/go-git"): [163287, 163314, 163330, 163328, 163331, 163329],
            (41470, "src-d/go-git"): [163313],
            (41469, "src-d/go-git"): [163309, 163307, 163308, 163316, 163255],
            (41468, "src-d/go-git"): [163312],
            (41467, "src-d/go-git"): [163278, 163289, 163290, 163293, 163306, 163281, 163282,
                                      163292,
                                      163285, 163311, 163277, 163295, 163310, 163276, 163304,
                                      163280,
                                      163286],
            (41485, "src-d/go-git"): [163257, 163158, 163298, 163200, 163246, 163302, 163199,
                                      163301,
                                      163254, 163299, 163300, 163259, 163262, 163303, 163256,
                                      163272,
                                      163273, 163297, 163253],
            (41484, "src-d/go-git"): [163245, 163269, 163261, 163243, 163266, 163265, 163247,
                                      163264,
                                      163263, 163222, 163242, 163268],
            (41483, "src-d/go-git"): [163248, 163250, 163235, 163239, 163211, 163252, 163234,
                                      163249,
                                      163223, 163233, 163241, 163236, 163240, 163251],
            (41482, "src-d/go-git"): [163231, 163218, 163219, 163216, 163232, 163215, 163213],
            (41481, "src-d/go-git"): [163230, 163226, 163225, 163229, 163224, 163228],
            (41480, "src-d/go-git"): [163198, 163201, 163208],
            (41479, "src-d/go-git"): [163205, 163206, 163207, 163167],
            (41478, "src-d/go-git"): [163178, 163203, 163202],
            (41477, "src-d/go-git"): [163176, 163179, 163184, 163182, 163175, 163183, 163185],
            (41476, "src-d/go-git"): [163187, 163188, 163190, 163192, 163193, 163160, 163189],
            (41517, "src-d/go-git"): [163161, 163159, 163166, 163165, 163157, 163174, 163162,
                                      163164],
            (41519, "src-d/go-git"): [163169, 163170, 163172, 163173],
            (41518, "src-d/go-git"): [163054, 163052, 163104, 163113, 163137, 163152, 163056,
                                      163127,
                                      163103, 163125, 163096, 163149, 163134, 163116, 163114,
                                      163118,
                                      163108, 163055, 163098, 163147, 163138, 163131, 163112,
                                      163136,
                                      163110, 163156, 163133, 163101, 163154, 163150, 163155,
                                      163126,
                                      163146, 163140, 163124, 163153, 163168, 163120, 163115,
                                      163122,
                                      163107, 163105, 163119, 163099, 163144, 163132, 163109,
                                      163057,
                                      163141, 163121, 163106, 163058, 163027, 163143, 163100,
                                      163151,
                                      163135, 163128, 163117, 163097, 163142, 163123],
            (41514, "src-d/go-git"): [163048, 163049, 163046, 163051, 163053],
            (41513, "src-d/go-git"): [163033, 163039, 163093, 163091, 163042, 163044, 163030,
                                      163092,
                                      163090, 163029, 163038, 163032, 163076, 163037, 163028,
                                      163094,
                                      163095, 163043, 163034, 163045, 163031, 163036],
            (41516, "src-d/go-git"): [163069, 163087, 163082, 163066, 163074, 163085, 163068,
                                      163081,
                                      163086, 163080, 163063, 163088, 163072, 163067, 163089,
                                      163075,
                                      163070, 163083, 163084],
            (41515, "src-d/go-git"): [163059, 163008, 162994, 163011, 163064, 163012, 162988,
                                      163013,
                                      163016, 163010, 163015, 163061, 163060, 163006, 163007,
                                      163005,
                                      162996, 163062, 163009],
            (41512, "src-d/go-git"): [163001, 163023, 162997, 162993, 163003, 162975, 163026,
                                      163002,
                                      163000, 162998, 163025, 162989, 162992, 162995],
            (41511, "src-d/go-git"): [162955, 162981, 162978, 162977, 162956, 162967, 162973,
                                      162890,
                                      162918, 162962, 162900, 162961, 162887, 162923, 163019,
                                      162970,
                                      162919, 162896, 162913, 162916, 162974, 162927, 162924,
                                      162966,
                                      162899, 162969, 163018, 162892, 162968, 162908, 162920,
                                      162903,
                                      162983, 162904, 162893, 162963, 163529, 162952, 162960,
                                      162912,
                                      162922, 162889, 162976, 162958, 163022, 162986, 162971,
                                      162959,
                                      162898, 162985, 162965, 162914, 162894, 162957, 162905,
                                      162911,
                                      162926, 163020, 162982, 162888, 162915, 162921, 162987,
                                      162895,
                                      162897, 162901, 162984, 162910, 162906, 162925, 162972],
            (41510, "src-d/go-git"): [],
            (41509, "src-d/go-git"): [], (41508, "src-d/go-git"): [], (41506, "src-d/go-git"): [],
            (41505, "src-d/go-git"): [], (41503, "src-d/go-git"): [], (41502, "src-d/go-git"): [],
            (41501, "src-d/go-git"): [], (41507, "src-d/go-git"): [], (41496, "src-d/go-git"): [],
            (41495, "src-d/go-git"): [], (41500, "src-d/go-git"): [], (41499, "src-d/go-git"): [],
            (41498, "src-d/go-git"): [], (41497, "src-d/go-git"): [], (41490, "src-d/go-git"): [],
            (41488, "src-d/go-git"): [], (41487, "src-d/go-git"): [], (41486, "src-d/go-git"): [],
            (41494, "src-d/go-git"): [], (41493, "src-d/go-git"): [], (41492, "src-d/go-git"): [],
            (41491, "src-d/go-git"): []},
        "prs_number": {
            (41475, "src-d/go-git"): [1203],
            (41474, "src-d/go-git"): [1165, 1181, 1179, 1197, 1200, 1175, 1199],
            (41473, "src-d/go-git"): [
                1160, 1097, 1145, 1126, 1119, 1118, 1131, 1096, 1090, 1146, 1154,
                1142, 1127, 1164, 1123, 1159, 1136, 1128, 1134, 1121, 1133, 1130,
                1132, 1124, 1125],
            (41472, "src-d/go-git"): [
                1080, 1070, 1095, 1076, 1088, 1092, 1115, 1081, 1072, 1099, 1084,
                1093, 1094, 1112, 1073],
            (41471, "src-d/go-git"): [1006, 1060, 1066, 1064, 1067, 1065],
            (41470, "src-d/go-git"): [1056],
            (41469, "src-d/go-git"): [1037, 1031, 1036, 1045, 963],
            (41468, "src-d/go-git"): [1028],
            (41467, "src-d/go-git"): [
                994, 1008, 1009, 1015, 989, 1000, 1001, 1013, 1004, 1025, 992,
                1019, 1022, 990, 987, 998, 1005],
            (41485, "src-d/go-git"): [
                968, 727, 978, 830, 928, 985, 828, 984, 962, 979, 982, 974, 932,
                986, 966, 950, 949, 977, 958],
            (41484, "src-d/go-git"): [927, 942, 929, 924, 939, 937, 887, 935, 933, 882, 921, 941],
            (41483, "src-d/go-git"): [
                888, 896, 905, 910, 862, 899, 904, 892, 885, 902, 920, 906, 916, 898],
            (41482, "src-d/go-git"): [857, 873, 874, 870, 859, 869, 864],
            (41481, "src-d/go-git"): [856, 848, 846, 855, 845, 854],
            (41480, "src-d/go-git"): [825, 833, 815],
            (41479, "src-d/go-git"): [807, 808, 810, 706],
            (41478, "src-d/go-git"): [784, 804, 803],
            (41477, "src-d/go-git"): [783, 786, 795, 792, 778, 794, 797],
            (41476, "src-d/go-git"): [754, 759, 766, 769, 771, 739, 762],
            (41517, "src-d/go-git"): [740, 731, 751, 749, 724, 721, 742, 744],
            (41519, "src-d/go-git"): [712, 714, 716, 720],
            (41518, "src-d/go-git"): [
                579, 577, 631, 652, 687, 646, 582, 668, 626, 666, 609, 640, 677,
                656, 653, 658, 586, 580, 613, 638, 688, 672, 608, 686, 588, 651,
                675, 617, 649, 641, 650, 667, 700, 690, 665, 647, 710, 660, 655,
                663, 585, 632, 659, 616, 698, 674, 587, 583, 695, 661, 633, 584,
                534, 697, 615, 643, 680, 669, 657, 610, 696, 664],
            (41514, "src-d/go-git"): [572, 573, 569, 576, 578],
            (41513, "src-d/go-git"): [
                544, 558, 531, 526, 563, 565, 540, 527, 522, 538, 555, 543, 501,
                554, 536, 532, 533, 564, 546, 567, 542, 552],
            (41516, "src-d/go-git"): [
                493, 515, 510, 487, 499, 513, 491, 509, 514, 507, 484, 516, 496,
                489, 517, 498, 492, 511, 512],
            (41515, "src-d/go-git"): [
                477, 464, 423, 469, 485, 472, 414, 473, 476, 467, 475, 479, 478,
                452, 453, 451, 432, 480, 465],
            (41512, "src-d/go-git"): [
                441, 405, 434, 422, 444, 369, 411, 442, 440, 436, 409, 418, 421, 429],
            (41511, "src-d/go-git"): [
                333, 384, 381, 375, 331, 360, 366, 252, 292, 346, 269, 344, 248,
                297, 395, 364, 293, 265, 286, 289, 368, 303, 299, 356, 268, 363,
                394, 258, 361, 282, 294, 273, 305, 276, 259, 347, 139, 324, 339,
                285, 296, 251, 374, 336, 400, 319, 365, 337, 267, 316, 355, 287,
                262, 334, 277, 284, 302, 398, 388, 250, 288, 295, 320, 263, 266,
                270, 314, 283, 278, 300, 367],
            (41510, "src-d/go-git"): [], (41509, "src-d/go-git"): [], (41508, "src-d/go-git"): [],
            (41506, "src-d/go-git"): [], (41505, "src-d/go-git"): [], (41503, "src-d/go-git"): [],
            (41502, "src-d/go-git"): [], (41501, "src-d/go-git"): [], (41507, "src-d/go-git"): [],
            (41496, "src-d/go-git"): [],
            (41495, "src-d/go-git"): [], (41500, "src-d/go-git"): [], (41499, "src-d/go-git"): [],
            (41498, "src-d/go-git"): [], (41497, "src-d/go-git"): [], (41490, "src-d/go-git"): [],
            (41488, "src-d/go-git"): [],
            (41487, "src-d/go-git"): [], (41486, "src-d/go-git"): [], (41494, "src-d/go-git"): [],
            (41493, "src-d/go-git"): [], (41492, "src-d/go-git"): [], (41491, "src-d/go-git"): []},
        "prs_additions": {
            (41475, "src-d/go-git"): [1], (41474, "src-d/go-git"): [129, 3, 83, 5, 30, 1, 40],
            (41473, "src-d/go-git"): [62, 971, 47, 46, 26, 73, 41, 256, 68, 42, 26,
                                      82, 57, 1, 143, 10, 28, 689, 61, 126, 4, 1043,
                                      961, 143, 34],
            (41472, "src-d/go-git"): [17, 33, 4, 19, 87, 1, 40, 0, 1, 1, 17, 67, 47, 1, 1],
            (41471, "src-d/go-git"): [1161, 27, 104, 7, 85, 62], (41470, "src-d/go-git"): [21],
            (41469, "src-d/go-git"): [1, 25, 73, 357, 126], (41468, "src-d/go-git"): [57],
            (41467, "src-d/go-git"): [95, 156, 8, 45, 1, 2, 50, 11, 17, 122, 27, 61, 0,
                                      128, 30, 55, 3],
            (41485, "src-d/go-git"): [15, 8983, 1, 54, 662, 3, 19, 9, 64, 271, 168,
                                      1, 43, 42, 2, 69, 111, 14, 66],
            (41484, "src-d/go-git"): [85, 131, 41, 9, 6, 7, 101, 36, 98, 68, 45, 260],
            (41483, "src-d/go-git"): [83, 484, 3, 32, 2, 725, 132, 45, 8, 49, 178,
                                      2949, 276, 831],
            (41482, "src-d/go-git"): [9, 2, 297, 49, 39, 12, 82],
            (41481, "src-d/go-git"): [19, 73, 67, 26, 4, 9],
            (41480, "src-d/go-git"): [285, 22, 62],
            (41479, "src-d/go-git"): [23, 1, 132, 33], (41478, "src-d/go-git"): [73, 0, 507],
            (41477, "src-d/go-git"): [26, 48, 10, 172, 39, 59, 19],
            (41476, "src-d/go-git"): [7, 9, 14, 61, 329, 779, 12],
            (41517, "src-d/go-git"): [1, 56, 56, 43, 106, 25, 2, 18],
            (41519, "src-d/go-git"): [19, 56, 19, 24],
            (41518, "src-d/go-git"): [38, 43, 13, 39, 110, 29, 367, 6, 36, 76, 70, 40, 51,
                                      3, 95, 172, 45, 73, 167, 1, 1, 22, 23, 324, 84, 29,
                                      67, 1, 59, 1, 278, 56, 142, 14, 205, 20, 1, 82, 84,
                                      148, 1, 6, 2, 69, 113, 1, 146, 1, 124, 23, 2, 3,
                                      24, 152, 4, 1, 336, 928, 3, 97, 4, 1],
            (41514, "src-d/go-git"): [277, 118, 15, 57, 33],
            (41513, "src-d/go-git"): [39, 62, 37, 61, 76, 23, 6, 75, 11, 20, 48, 22, 219,
                                      2, 33, 44, 15, 43, 61, 48, 12, 56],
            (41516, "src-d/go-git"): [95, 493, 318, 116, 271, 91, 118, 300, 154, 387, 46, 28, 18,
                                      166, 4, 152, 85, 109, 123],
            (41515, "src-d/go-git"): [0, 31, 36, 76, 205, 35, 2, 3, 49, 95, 42, 32, 61,
                                      1, 1, 6, 266, 51, 23],
            (41512, "src-d/go-git"): [56, 15, 69, 7, 35, 25, 35, 6, 24, 198, 37, 189, 11,
                                      758],
            (41511, "src-d/go-git"): [22, 117, 68, 559, 45, 22, 151, 873, 144, 93, 206,
                                      483, 637, 784, 206, 38, 29, 4, 25, 21, 114, 483,
                                      5, 41, 12, 142, 20, 71, 205, 26, 22, 1269, 59,
                                      302, 110, 68, 1546, 62, 1288, 72, 426, 19, 10, 14,
                                      94, 30, 201, 331, 4, 1, 35, 1, 1246, 14, 0,
                                      67, 26, 74, 1614, 109, 56, 52, 6, 52, 12, 1092,
                                      36, 3, 1, 31, 45],
            (41510, "src-d/go-git"): [], (41509, "src-d/go-git"): [], (41508, "src-d/go-git"): [],
            (41506, "src-d/go-git"): [], (41505, "src-d/go-git"): [], (41503, "src-d/go-git"): [],
            (41502, "src-d/go-git"): [], (41501, "src-d/go-git"): [], (41507, "src-d/go-git"): [],
            (41496, "src-d/go-git"): [], (41495, "src-d/go-git"): [], (41500, "src-d/go-git"): [],
            (41499, "src-d/go-git"): [], (41498, "src-d/go-git"): [], (41497, "src-d/go-git"): [],
            (41490, "src-d/go-git"): [], (41488, "src-d/go-git"): [], (41487, "src-d/go-git"): [],
            (41486, "src-d/go-git"): [], (41494, "src-d/go-git"): [], (41493, "src-d/go-git"): [],
            (41492, "src-d/go-git"): [], (41491, "src-d/go-git"): []},
        "prs_deletions": {
            (41475, "src-d/go-git"): [1],
            (41474, "src-d/go-git"): [16, 3, 18, 5, 10, 1, 49],
            (41473, "src-d/go-git"): [11, 0, 0, 19, 4, 1, 33, 7, 2, 49, 2, 37, 4,
                                      1, 49, 11, 51, 0, 63, 56, 6, 0, 6, 105, 9],
            (41472, "src-d/go-git"): [1, 1, 1, 1, 2, 1, 15, 11, 1, 1, 1, 9, 0, 1, 1],
            (41471, "src-d/go-git"): [8, 1, 17, 6, 11, 14],
            (41470, "src-d/go-git"): [5],
            (41469, "src-d/go-git"): [5, 1, 50, 32, 61],
            (41468, "src-d/go-git"): [19],
            (41467, "src-d/go-git"): [47, 6, 1, 15, 1, 1, 0, 3, 17, 4, 1, 0, 56, 0, 2, 0, 1],
            (41485, "src-d/go-git"): [5, 2116, 1, 11, 69, 2, 5, 14, 12, 6, 0,
                                      1, 2, 5, 2, 7, 165, 1, 24],
            (41484, "src-d/go-git"): [1, 2, 2, 0, 2, 0, 3, 18, 34, 0, 5, 9],
            (41483, "src-d/go-git"): [0, 387, 2, 6, 1, 173, 32, 18, 3, 135, 7,
                                      1539, 53, 47],
            (41482, "src-d/go-git"): [8, 2, 9, 5, 1, 2, 94],
            (41481, "src-d/go-git"): [4, 12, 23, 1, 4, 10],
            (41480, "src-d/go-git"): [7, 2, 8],
            (41479, "src-d/go-git"): [3, 0, 2, 8],
            (41478, "src-d/go-git"): [6, 0, 7],
            (41477, "src-d/go-git"): [1, 0, 2, 9, 0, 39, 2],
            (41476, "src-d/go-git"): [2, 0, 83, 22, 3, 38, 2],
            (41517, "src-d/go-git"): [1, 2, 10, 2, 13, 1, 2, 8],
            (41519, "src-d/go-git"): [4, 12, 13, 6],
            (41518, "src-d/go-git"): [11, 0, 4, 2, 0, 4, 56, 7, 11, 6, 0, 0, 5,
                                      2, 0, 7, 15, 40, 0, 1, 1, 22, 3, 1, 3, 48,
                                      0, 1, 55, 0, 269, 118, 33, 18, 22, 20, 1,
                                      54, 89,
                                      1, 1, 0, 2, 0, 68, 1, 42, 1, 35, 2, 2, 17,
                                      0, 29, 4, 1, 19, 74, 1, 4, 11, 1],
            (41514, "src-d/go-git"): [130, 6, 10, 13, 6],
            (41513, "src-d/go-git"): [0, 12, 39, 2, 24, 1, 0, 2, 8, 6, 0, 1, 81, 2, 1, 15, 7,
                                      11, 1, 3, 4, 38],
            (41516, "src-d/go-git"): [51, 93, 138, 23, 271, 89, 47, 66, 113, 42, 1, 1, 0,
                                      41, 3, 171, 16, 6, 2],
            (41515, "src-d/go-git"): [5, 21, 3, 17, 109, 41, 2, 2, 7, 12, 23, 3, 36,
                                      1, 1, 10, 77, 16, 18],
            (41512, "src-d/go-git"): [115, 14, 25, 5, 7, 31, 1, 6, 1, 27, 15, 39, 11,
                                      0],
            (41511, "src-d/go-git"): [6, 0, 0, 54, 3, 21, 22, 0, 98, 2, 14,
                                      58, 834, 283, 2, 37, 2, 18, 1, 16, 52, 483,
                                      5, 13, 1, 9, 4, 10, 4, 3, 16, 536, 7,
                                      81, 56, 23, 0, 0, 292, 9, 43, 17, 0, 8,
                                      441, 7, 62, 173, 45, 1, 9, 0, 13, 14, 1900,
                                      66, 2, 0, 1, 2, 11, 42, 1, 54, 5, 131,
                                      2, 3, 1, 51, 1],
            (41510, "src-d/go-git"): [], (41509, "src-d/go-git"): [], (41508, "src-d/go-git"): [],
            (41506, "src-d/go-git"): [], (41505, "src-d/go-git"): [], (41503, "src-d/go-git"): [],
            (41502, "src-d/go-git"): [], (41501, "src-d/go-git"): [], (41507, "src-d/go-git"): [],
            (41496, "src-d/go-git"): [], (41495, "src-d/go-git"): [], (41500, "src-d/go-git"): [],
            (41499, "src-d/go-git"): [], (41498, "src-d/go-git"): [], (41497, "src-d/go-git"): [],
            (41490, "src-d/go-git"): [], (41488, "src-d/go-git"): [], (41487, "src-d/go-git"): [],
            (41486, "src-d/go-git"): [], (41494, "src-d/go-git"): [], (41493, "src-d/go-git"): [],
            (41492, "src-d/go-git"): [], (41491, "src-d/go-git"): []},
        "prs_user_node_id": {
            (41475, "src-d/go-git"): [40187],
            (41474, "src-d/go-git"): [40292, 39771, 39887, 40152, 39789, 40025, 39789],
            (41473, "src-d/go-git"): [40410, 39974, 40146, 39868, 39828, 39828, 39789, 39974,
                                      39890,
                                      40410, 39994, 40114, 40187, 40158, 39868, 40083, 39828,
                                      39828,
                                      39828, 39828, 39828, 39868, 39828, 39868, 39868],
            (41472, "src-d/go-git"): [39849, 40375, 40298, 39849, 39982, 40298, 39789, 40021,
                                      40243,
                                      39974, 40020, 39849, 40298, 40138, 40243],
            (41471, "src-d/go-git"): [39789, 40374, 40374, 40039, 39926, 40020],
            (41470, "src-d/go-git"): [39849],
            (41469, "src-d/go-git"): [40328, 39849, 39849, 40189, 39828],
            (41468, "src-d/go-git"): [40070],
            (41467, "src-d/go-git"): [39913, 40070, 40070, 40414, 40031, 40070, 40044, 40228,
                                      39798,
                                      39789, 40261, 39913, 39926, 40228, 40061, 39789, 40044],
            (41485, "src-d/go-git"): [40070, 39968, 40377, 40410, 39900, 39789, 40186, 39789,
                                      39849,
                                      40181, 40374, 39764, 39916, 39789, 40095, 40189, 40189,
                                      39828, 40189],
            (41484, "src-d/go-git"): [39849, 39849, 39849, 40414, 40197, 40076, 39757, 39849,
                                      40189,
                                      39863, 39849, 39849],
            (41483, "src-d/go-git"): [40345, 39957, 39926, 40414, 40165, 39957, 39849, 39849,
                                      40266,
                                      39849, 39900, 39957, 39849, 39849],
            (41482, "src-d/go-git"): [39957, 40070, 40070, 39789, 40107, 40135, 40070],
            (41481, "src-d/go-git"): [40189, 39788, 40135, 39849, 40135, 39926],
            (41480, "src-d/go-git"): [39804, 39849, 40189],
            (41479, "src-d/go-git"): [40374, 39849, 39849, 40175],
            (41478, "src-d/go-git"): [40093, 40068, 39845],
            (41477, "src-d/go-git"): [39804, 39789, 39881, 39926, 39896, 39849, 39788],
            (41476, "src-d/go-git"): [39881, 40287, 40093, 40294, 39874, 39789, 40038],
            (41517, "src-d/go-git"): [40283, 39849, 39789, 0, 39849, 40090, 40288, 39849],
            (41519, "src-d/go-git"): [39849, 39800, 39849, 39849],
            (41518, "src-d/go-git"): [39957, 40059, 40374, 40283, 39980, 39789, 39957, 39936,
                                      40239,
                                      40374, 40319, 39962, 39968, 40319, 40319, 40319, 40374,
                                      39957,
                                      40319, 40051, 40319, 40283, 40052, 40319, 39957, 39957,
                                      40319,
                                      40109, 39789, 39844, 39789, 40283, 39849, 39789, 40374,
                                      40283,
                                      40392, 40175, 39789, 40319, 40230, 40374, 40283, 40319,
                                      39849,
                                      40283, 40374, 40246, 40319, 40283, 40170, 39957, 40293,
                                      39926,
                                      40109, 39989, 39789, 40374, 39926, 40374, 40283, 40239],
            (41514, "src-d/go-git"): [39789, 40239, 39957, 39789, 39957],
            (41513, "src-d/go-git"): [39957, 40239, 39789, 40123, 39957, 40374, 39818, 40374,
                                      39799,
                                      40421, 39789, 39957, 40070, 40374, 39818, 39789, 39818,
                                      40374,
                                      40239, 40374, 39818, 39818],
            (41516, "src-d/go-git"): [39789, 40070, 40070, 39789, 39789, 39789, 40070, 39789,
                                      40070,
                                      39789, 40239, 40070, 39912, 39789, 40070, 39789, 39926,
                                      39789, 39789],
            (41515, "src-d/go-git"): [39891, 40070, 40070, 39789, 39789, 40070, 40045, 39891,
                                      40070,
                                      40070, 39789, 39789, 39789, 39891, 40070, 40070, 39926,
                                      39789, 40070],
            (41512, "src-d/go-git"): [39789, 40317, 40239, 40070, 40336, 40070, 39926, 39926,
                                      39926,
                                      39789, 40070, 40070, 40070, 40336],
            (41511, "src-d/go-git"): [39891, 40070, 39789, 39789, 39800, 40070, 40070, 40418,
                                      39926,
                                      39789, 40070, 39789, 39789, 40418, 39789, 39789, 39926,
                                      40418,
                                      39926, 39926, 40070, 39926, 39926, 39789, 40418, 40070,
                                      39926,
                                      39926, 39789, 39789, 39926, 40418, 40093, 40418, 40070,
                                      39789,
                                      40175, 39926, 39789, 40418, 39926, 40385, 39926, 40073,
                                      39926,
                                      39926, 40070, 39926, 40418, 39906, 39789, 39926, 40418,
                                      39891,
                                      39926, 39926, 40284, 39789, 39926, 39789, 39926, 39926,
                                      39912,
                                      40070, 40418, 39789, 39912, 39789, 39926, 40418, 40070],
            (41510, "src-d/go-git"): [], (41509, "src-d/go-git"): [],
            (41508, "src-d/go-git"): [], (41506, "src-d/go-git"): [],
            (41505, "src-d/go-git"): [], (41503, "src-d/go-git"): [],
            (41502, "src-d/go-git"): [], (41501, "src-d/go-git"): [],
            (41507, "src-d/go-git"): [], (41496, "src-d/go-git"): [],
            (41495, "src-d/go-git"): [], (41500, "src-d/go-git"): [],
            (41499, "src-d/go-git"): [], (41498, "src-d/go-git"): [],
            (41497, "src-d/go-git"): [], (41490, "src-d/go-git"): [],
            (41488, "src-d/go-git"): [], (41487, "src-d/go-git"): [],
            (41486, "src-d/go-git"): [], (41494, "src-d/go-git"): [],
            (41493, "src-d/go-git"): [], (41492, "src-d/go-git"): [],
            (41491, "src-d/go-git"): []},
        "prs_title": {
            (41475, "src-d/go-git"): [
                "worktree: force convert to int64 to support 32bit os. Fix #1202"],
            (41474, "src-d/go-git"): ["Remote: add Prune option to PushOptions",
                                      "*: fix typos in comments",
                                      "Worktree: improve build index performance.",
                                      "Make http.AuthMethod setAuth public. Fixes #1196",
                                      "*: go module update", "config: added missing dot.",
                                      "*: code quality improvements"],
            (41473, "src-d/go-git"): [
                "fix wildcard handling in RefSpec matching",
                "Create merge-base feature",
                "Worktree: keep local changes when checkout branch",
                "plumbing: format/index perf, buffered reads, reflection removal",
                "plumbing: idxfile, avoid unnecessary building of reverse offset/hash map",
                "plumbing: object, Fix tag message decoding",
                "go modules update",
                "Add merge base command",
                "ssh: leverage proxy from environment",
                "improve ResolveRevision''''s Ref lookup path",
                "Support the ''''rebase'''' config key for branches",
                "git : allows to create a Remote without a Repository",
                "plumbing: object/{commit,tag} add EncodeWithoutSignature, Implement #1116",
                "use constant instead of literal string",
                "filesystem: ObjectStorage, MaxOpenDescriptors option",
                "plumbing: format/packfile, Fix data race and resource leak.",
                "plumbing: format/idxfile, avoid creating temporary buffers to decode "
                "integers",
                "plumbing: format/commitgraph, add APIs for reading and writing commit-graph "
                "files",
                "plumbing: format/commitgraph, rename structs/fields to follow the terms "
                "used by git more closely",
                "plumbing: packfile, apply small object reading optimization also for delta "
                "objects",
                "plumbing: format/commitgraph, clean up error handling",
                "plumbing: format/gitattributes support",
                "plumbing: object, add APIs for traversing over commit graphs",
                "plumbing: packfile/scanner, readability/performance improvements, zlib "
                "pooling",
                "plumbing: TreeWalker performance improvement, bufio pool for objects"],
            (41472, "src-d/go-git"): [
                "git: fix goroutine block while pushing a remote",
                "worktree: allow manual ignore patterns when no .gitignore is available",
                "fix panic in object.Tree.FindEntry",
                "plumbing/cache: check for empty cache list",
                "plumbing: object, Count stats properly when no new line added at the …",
                "fix missing error in bfsCommitIterator",
                "plumbing: commit.StatsContext and fix for orphan commit",
                "git: remove potentially duplicate check for unstaged files",
                "git: Fix typo",
                "travis: drop go1.10 add go1.12",
                "Increase diffmatchcpatch timeout",
                "transactional: implement storer.PackfileWriter",
                "add Repository.CreateRemoteAnonymous",
                "examples: commit, fixed minor typo in info statement",
                "git: Fix typo"],
            (41471, "src-d/go-git"): [
                "storage: transactional, new storage with transactional capabilities",
                "packfile: get object size correctly for delta objects",
                'remote: speed up pushes when the "remote" repo is local',
                "worktree: add sentinel error for non-fast-forward pull",
                "Ignore missing references/objects on log --all",
                "Remove Unicode normalization in difftree"],
            (41470, "src-d/go-git"): ["storage/filesystem: check file object before using cache"],
            (41469, "src-d/go-git"): [
                "Simplify return statement in receivePackNoCheck",
                "git: return better error message when packfile cannot be downloaded",
                "storage/dotgit: use fs capabilities in setRef",
                "Implement git log --all",
                "plumbing: format/packfile, performance optimizations for reading large "
                "commit histories"],
            (41468, "src-d/go-git"): ["repository: fix plain clone error handling regression"],
            (41467, "src-d/go-git"): [
                'plumbing/format/packfile: Fix broken "thin" packfile support. Fixes #991',
                "cleanup after failed clone", "http: improve TokenAuth documentation",
                "repository: Fix RefSpec for a single tag.", "README: Fixed a typo.",
                "add StackOverflow to support channels",
                "plumbing: transport/http, Add missing host/port on redirect. Fixes #820",
                " plumbing: ssh, Fix flaky test TestAdvertisedReferencesNotExists. Fixes #969",
                "Fix spelling and grammar in docs and example",
                "plumbing: format/index: support for EOIE extension",
                "git: enables building on OpenBSD, Dragonfly BSD and Solaris",
                "storage/filesystem: Added reindex method to  reindex packfiles",
                "plumbing: format/packfile, remove  unused getObjectData method",
                "examples: PlainClone with Basic Authentication (Password & Access Token)",
                "remote: use reference deltas on push when the remote server does not …",
                "plumbing: ReferenceName constructors", "update gcfg dependency to v1.4.0"],
            (41485, "src-d/go-git"): [
                "test: improve test for urlencoded user:pass",
                "references: sort: compare author timestamps when commit timestamps are equal."
                " Fixes #725", "git: Fix Status.IsClean() documentation",
                "Teach ResolveRevision how to look up annotated tags",
                "git: Add tagging support",
                "repository: improve CheckoutOption.Hash doc",
                "Use remote name in fetch while clone",
                "repository: allow open non-bare repositories as bare",
                "storage/filesystem: keep packs open in PackfileIter",
                "plumbing: object, Add support for Log with filenames. Fixes #826",
                "tree: add a Size() method for getting plaintext size",
                "use time.IsZero in Prune",
                "Fix `fatal: corrupt patch` error in unified diff format",
                "blame: fix edge case with missing \n in content length causing mismatched "
                "length error",
                "all: remove extra ''''s'''' in \"mismatch\"",
                "plumbing/transport: ssh check if list of known_hosts files is empty.",
                "Expose Storage cache.",
                "config: Add test for Windows local paths.",
                "Fix potential LRU cache size issue."],
            (41484, "src-d/go-git"): [
                "plumbing/idxfile: object iterators returns entries in offset order",
                "storage/dotgit: add KeepDescriptors option",
                "plumbing, storage: add bases to the common cache",
                "plumbing/format: gitignore, fixed an edge case for .gitignore",
                "Clamp object timestamps before unix epoch to unix epoch",
                "config: add commentChar to core config struct",
                "added support for quarantine directory",
                "storage/dotgit: search for incoming dir only once",
                "Remove empty dirs when cleaning with Dir opt.",
                "worktree: sort the tree object.  Fixes #881",
                "plumbing/object: fix panic when reading object header",
                "plumbing/storer: add ExclusiveAccess option to Storer"],
            (41483, "src-d/go-git"): [
                "plumbing/transport/internal: common, support Gogs for ErrRepositoryNotFound",
                "plumbing/format/idxfile: add new Index and MemoryIndex",
                "Fix wrong godoc on Tags() method.",
                "Fixed cloning of a single tag",
                "git: fix documentation for Notes",
                " plumbing: packfile, new Packfile representation",
                "Tests and indexes in packfile decoder",
                "plumbing/object: fix pgp signature encoder/decoder",
                "plumbing: object, return ErrFileNotFound in FindEntry. Fixes #883",
                "Bugfixes and IndexStorage",
                "git: Add ability to PGP sign commits",
                "Improve packfile reading performance",
                "Improvement/memory consumption new packfile parser",
                "Feature/new packfile parser"],
            (41482, "src-d/go-git"): [
                "storage: filesystem, make ObjectStorage constructor public",
                "utils: diff, skip useless rune->string conversion",
                "plumbing: add context to allow cancel on diff/patch computing",
                "Remote.Fetch: error on missing remote reference",
                "plumbing/transport: http, Adds token authentication support [Fixes #858]",
                "packfile: optimise NewIndexFromIdxFile for a very common case",
                "storage/filesystem: avoid norwfs build flag"],
            (41481, "src-d/go-git"): [
                "plumbing: packfile, Don''''t copy empty objects. Fixes #840",
                "config: modules, worktree: Submodule fixes for CVE-2018-11235",
                "packfile: improve Index memory representation to be more compact",
                "plumbing: object, adds tree path cache to trees. Fixes #793",
                "idxfile: optimise allocations in readObjectNames",
                "dotgit: Move package outside internal."],
            (41480, "src-d/go-git"): [
                "Worktree: Provide ability to add excludes to worktree",
                "git: remote, Do not iterate all references on update.",
                'Fix for "Worktree Add function adds ".git" directory"'],
            (41479, "src-d/go-git"): [
                "dotgit: ignore filenames that don''''t match a hash",
                "storage: dotgit, init fixtures in benchmark. Fixes #770",
                "git: remote, Add shallow commits instead of substituting. Fixes #412",
                "Resolve full commit sha"],
            (41478, "src-d/go-git"): [
                "add PlainOpen variant to find .git in parent dirs",
                "use bsd superset for conditional compilation",
                "config: adds branches to config for tracking branches against remotes…"],
            (41477, "src-d/go-git"): [
                "Fix RefSpec.Src()", "*: skip time consuming tests",
                "Add commit hash to blame result",
                "Resolve HEAD if symRefs capability is not supported",
                "Worktree.Checkout: handling of symlink on Windows",
                "Use CheckClose with named returns and fix tests",
                "plumbing: format: pktline, Accept oversized pkt-lines up to 65524 bytes"],
            (41476, "src-d/go-git"): [
                "blame: Add blame line data",
                "plumbing: ssh, return error when creating public keys from invalid PEM",
                "Unused params, unused code, make Go tip''''s vet happy",
                "storage/filesystem: optimize packfile iterator",
                "repository.Log: add alternatives for commit traversal order",
                "new methods Worktree.[AddGlob|RemoveBlob] and recursive "
                "Worktree.[Add|Remove]",
                "plubming: transport, Escape the user and pswd for endpoint. Fixes #723"],
            (41517, "src-d/go-git"): [
                "storage/filesystem/shallow: fix error checking",
                "plumbing: format/packfile, fix crash with cycle deltas",
                "transport: http, fix services redirecting only info/refs",
                "plumbing: diff, fix crash when a small ending equal-chunk",
                "plumbing: packfile, Add a buffer to crc writer",
                "Support for clone without checkout (git clone -n)",
                "Fix mistyping",
                "plumbing: format/packfile, fix panic retrieving object hash."],
            (41519, "src-d/go-git"): [
                "Set default pack window size in config",
                "add branch add/remove example",
                "Clean reconstructed objects outside pack window",
                "plumbing: cache, modify cache to delete more than one item to free space"],
            (41518, "src-d/go-git"): [
                "revlist: do not visit again already visited parents",
                "Worktree.Add: Support Add deleted files, fixes #571",
                "packfile: use buffer pool for diffs",
                "plumbing/object: do not eat error on tree decode",
                "check .ssh/config for host and port overrides; fixes #629",
                "format: packfile fix DecodeObjectAt when Decoder has type",
                "packfile: improve performance of delta generation",
                "Updating the outdated README example to the new one",
                "packp/capability: Skip argument validations for unknown capabilities",
                "dotgit: handle refs that exist in both packed-refs and a loose ref file",
                "remote: add support for ls-remote",
                "utils: merkletrie, filesystem fix symlinks to dir",
                "object: patch, fix stats for submodules (fixes #654)",
                "plumbing: object, fix Commit.Verify test",
                "plumbing: object, new Commit.Verify method",
                "plumbing: object/tag, add signature and verification support",
                "plumbing: the commit walker can skip externally-seen commits",
                "remote: iterate over references only once",
                "Add Stats() to Commit",
                "Updating reference to the git object model",
                "doc: update compatibility for clean", "all: gofmt -s",
                "Add port to SCP Endpoints",
                "git: worktree, add Grep() method for git grep",
                "revlist: do not revisit ancestors as long as all branches are visited",
                "dotgit: remove ref cache for packed refs",
                "git: worktree, add Clean() method for git clean",
                "Fix spelling",
                "transport: made public all the fields and standardized AuthMethod",
                "fix: a range loop can break in advance",
                "transport: converts Endpoint interface into a struct",
                "all: simplification",
                "Add a setRef and rewritePackedRefsWhileLocked versions that supports non rw "
                "fs",
                "README.md update",
                "remote: support for non-force, fast-forward-only fetches",
                "examples,plumbing,utils: typo fixes", "fix typo",
                "fix Repository.ResolveRevision for branch and tag",
                "*: update to go-billy.v4 and go-git-fixtures.v3",
                "storage: filesystem, add support for git alternates",
                "examples: update link to GoDoc in _examples/storage",
                "packfile: delete index maps from memory when no longer needed",
                "doc: Update compatibility for commit/tag verify",
                "Add support for signed commits",
                "plumbing: cache, enforce the use of cache in packfile decoder",
                "dotgit: use Equal method of time.Time for equality",
                "config: support a configurable, and turn-off-able, pack.window",
                "Minor fix to grammatical error in error message for ErrRepositoryNotExists",
                "git: Worktree.Grep() support multiple patterns and pathspecs",
                "all: fixes for ineffective assign",
                "travis: update go versions",
                "revert: revlist: do not revisit already visited ancestors",
                "plumbing: object, commit.Parent() method",
                "plumbing: packafile, improve delta reutilization",
                "Fix spelling Unstagged -> Unstaged",
                "Fix typo in the readme",
                "License upgrade, plus code of conduct and contributing guidelines",
                "storage/repository: add new functions for garbage collection",
                "plumbing: transport/http, Close http.Body reader when needed",
                "remote: add the last 100 commits for each ref in haves list",
                "*: simplication",
                "plumbing/transport: Fix truncated comment in Endpoint"],
            (41514, "src-d/go-git"): [
                "Worktree.Reset refactor and Soft, Merge, Hard and Mixed modes",
                "Add sideband support for push",
                "dotgit: avoid duplicated references returned by Refs",
                "Repository.Clone added Tags option, and set by default AllTags",
                "packfile: improve performance a little by reducing gc pressure"],
            (41513, "src-d/go-git"): [
                "fix race condition on ObjectLRU",
                "repository: Resolve commit when cloning annotated tag, fixes #557",
                "plumbing: moved `Reference.Is*` methods to `ReferenceName.Is*`",
                "Normalize filenames before comparing.",
                "dotgit: rewrite the way references are looked up",
                "plumbing: use sliding window in delta calculations, like git CL",
                "*: windows support, skipped receive_pack_test for git transport",
                "plumbing: fix pack commands for the file client on Windows",
                "reuse Auth method when recursing submodules, fixes #521",
                "Avoid using user.Current()", "_examples: context",
                "prevent PackWriter from using Notify if nothing was written",
                "config: multiple values in RemoteConfig (URLs and Fetch)",
                "plumbing: use LookPath instead of Stat to fix Windows executables",
                "*: windows support, some more fixes (2)",
                "Remote.Clone fix clone of tags in shallow mode",
                "*: windows support, some more fixes",
                "plumbing: use `seen` map in tree walker",
                "examples: add example for pulling changes",
                "remote: avoid expensive revlist operation when only deleting refs",
                "serialized remotes in alphabetical order",
                "packp: fixed encoding when HEAD is not a valid ref"],
            (41516, "src-d/go-git"): [
                "*: several windows support fixes",
                " storage: reuse deltas from packfiles",
                "packfile: create packfile.Index and reuse it",
                "worktree: checkout, create branch",
                "move Repository.Pull to Worktree.Pull",
                "worktree: expose underlying filesystem",
                "*: add more IO error checks",
                "*: package context support in Repository, Remote and Submodule",
                "cache: reuse object cache for delta resolution, use LRU policy",
                "transport: context package support allowing cancellation of any network "
                "operation",
                "Add example code for listing tags",
                "revlist: ignore all objects reachable from ignored objects",
                "Implement a NoTags mode for fetch that mimics git fetch --no-tags",
                "repository: allow push from shallow repositories",
                "filesystem: reuse cache for packfile iterator",
                "remote: push, update remote refs on push",
                "packfile: Avoid panics patching corrupted deltas.",
                "remote: pull refactor to match default behavior of cgit",
                "format: idxfile, support for >2Gb packfiles"],
            (41515, "src-d/go-git"): [
                "git: remove ErrObjectNotFound in favor of plumbing.ErrObjectNotFound",
                "transport/file: avoid race with Command.Wait, fixes #463",
                "transport/ssh: allow passing SSH options",
                "plumbing: protocol, fix handling multiple ACK on upload-pack",
                "remote: fetch, correct behavior on tags", "improve delete support on push",
                "Fixed modules directory path", "Use buffered IO for decoding index files.",
                "transport/server: add asClient parameter", "remote: fix push delete, closes #466",
                "plumbing: protocol, fix handling multiple ACK on upload-pack and test…",
                "remote: avoid duplicate haves", "worktree: test improvemnts on empty worktree",
                "storage/filesystem: Fix nil dereference in Shallow()",
                "fix race on packfile writer, fixes #351",
                "capability: accept unknown capabilities, fixes #450", "transport: http push",
                "remote: fix Worktree.Status on empty repository", "fix reference shortening"],
            (41512, "src-d/go-git"): [
                "worktree: Add create and push the blob objects to the storer",
                "Support SSH Agent Auth on Windows",
                "Update local remote references during fetch even if no pack needs to be "
                "received",
                "fix gofmt",
                "Fixes checkout not possible with (untracked) files under gitignore",
                "Partial windows support",
                "packfile: A copy operation cannot be bigger than 64kb",
                "internal/dotgit: rewrite code to avoid stackoverflow errors",
                "revlist: ignore treeEntries that are submodules.",
                "worktree: symlink support",
                "storage/filesystem: call initialization explicitly, fixes #408",
                "fix push on git and ssh",
                "fix naming of NewCommit{Pre,Post}Iterator",
                "Adds .gitignore support"],
            (41511, "src-d/go-git"): [
                "Lazily load object index.",
                "README: add table with supported git features",
                "examples: commit example",
                "worktree: Commit method implementation",
                "add git checkout example + housekeeping",
                "fix go vet issues, add go vet to CI",
                "support force push (refspec with +)",
                "add merkletrie iterator and its helper frame type",
                "plumbing/revlist: input as a slice of hashes instead of commits",
                "transport: ssh, default HostKeyCallback",
                "plumbing/storer: add RemoveReference",
                "worktree: reset and checkout support for submodules",
                "package plumbing documentation improvements",
                "issue #274: new filemode package",
                "worktree: Remove and Move methods",
                "plumbing: index, Entries converted in a slice of pointers",
                "_examples: improve documentation (fix #238)",
                "simplify noder mocks in tests",
                "plumbing/storer: referenceIterator now returns the error if any",
                "plumbing/cache: specify units in memory size (Fix #234)",
                "do not convert local paths to URL",
                "project: move imports from srcd.works to gopkg.in",
                "Return values of Read not checked (fix #65)",
                "transport: ssh, new DefaultAuthBuilder variable",
                "merkletrie: fix const action type fuck up",
                "add support for .git as file, fixes #348",
                "transport/server: use Endpoint string representation as a map key.",
                "Fix missing objects if they where deltified using ref-delta",
                "worktree: add method",
                "plumbing/transport: git, error on empty SSH_AUTH_SOCK",
                "Remove TODOs from documentation",
                "difftree for git.Trees",
                "plumbing/object: add WalkCommitHistoryPost func",
                "Fix issue 275 (edited)",
                "Improve documentation",
                "transport: ssh, NewPublicKeys helper",
                "Add revision implementation",
                "object: fix Change.Files() method behavior (fix #317)",
                "worktree, status and reset implementation based on merkletrie",
                "Fix issue 279",
                "git: Repository methods changes",
                "Fix compile-time error on Windows",
                "format/packfile: fix bug when the delta depth is equals to 50",
                "plumbing: transport, handle 403 in http transport",
                "format/packfile: improve binary delta algorithm",
                "references.go: fix Parents from commit iterator",
                "transport: make Endpoint an interface, fixes #362",
                "Add Repository.Log() method (fix #298)",
                "transport/file: fix race condition on test",
                "Updated README in-memory example.",
                "transport: ssh, NewPublicKeys support for encrypted PEM files",
                "Add fast_finish flag to travis configuration",
                "add difftree for noders",
                "Export raw config",
                "cshared: remove directory (Fix #236)",
                "plumbing/object: move difftree to object package",
                "Support slash separated branch",
                "storage: filesystem, initialize the default folder scaffolding",
                "format/diff: unified diff encoder and public API",
                "examples: aerospike example",
                "plumbing: improve documentation (Fix #242)",
                "improve git package documentation (fix #231)",
                "Work around a Go bug when parsing timezones",
                "git: make Storer public in Repository.",
                "transport/file: delete suite tmp dir at teardown",
                "Submodules init and update",
                "plumbing: Use ReadBytes() rather than ReadSlice()",
                "travis update to 1.8 and makefile silence commands",
                "cache: move package to plumbing",
                "difftree: simplify hash comparison with deprecated files modes",
                "add test for tags push, closes #354"],
            (41510, "src-d/go-git"): [], (41509, "src-d/go-git"): [], (41508, "src-d/go-git"): [],
            (41506, "src-d/go-git"): [], (41505, "src-d/go-git"): [], (41503, "src-d/go-git"): [],
            (41502, "src-d/go-git"): [],
            (41501, "src-d/go-git"): [], (41507, "src-d/go-git"): [], (41496, "src-d/go-git"): [],
            (41495, "src-d/go-git"): [], (41500, "src-d/go-git"): [], (41499, "src-d/go-git"): [],
            (41498, "src-d/go-git"): [],
            (41497, "src-d/go-git"): [], (41490, "src-d/go-git"): [], (41488, "src-d/go-git"): [],
            (41487, "src-d/go-git"): [], (41486, "src-d/go-git"): [], (41494, "src-d/go-git"): [],
            (41493, "src-d/go-git"): [],
            (41492, "src-d/go-git"): [], (41491, "src-d/go-git"): []},
        "prs_jira": {
            (41475, "src-d/go-git"): None, (41474, "src-d/go-git"): None,
            (41473, "src-d/go-git"): None, (41472, "src-d/go-git"): None,
            (41471, "src-d/go-git"): None, (41470, "src-d/go-git"): None,
            (41469, "src-d/go-git"): None, (41468, "src-d/go-git"): None,
            (41467, "src-d/go-git"): None, (41485, "src-d/go-git"): None,
            (41484, "src-d/go-git"): None, (41483, "src-d/go-git"): None,
            (41482, "src-d/go-git"): None, (41481, "src-d/go-git"): None,
            (41480, "src-d/go-git"): None, (41479, "src-d/go-git"): None,
            (41478, "src-d/go-git"): None, (41477, "src-d/go-git"): None,
            (41476, "src-d/go-git"): None, (41517, "src-d/go-git"): None,
            (41519, "src-d/go-git"): None, (41518, "src-d/go-git"): None,
            (41514, "src-d/go-git"): None, (41513, "src-d/go-git"): None,
            (41516, "src-d/go-git"): None, (41515, "src-d/go-git"): None,
            (41512, "src-d/go-git"): None, (41511, "src-d/go-git"): None,
            (41510, "src-d/go-git"): None, (41509, "src-d/go-git"): None,
            (41508, "src-d/go-git"): None, (41506, "src-d/go-git"): None,
            (41505, "src-d/go-git"): None, (41503, "src-d/go-git"): None,
            (41502, "src-d/go-git"): None, (41501, "src-d/go-git"): None,
            (41507, "src-d/go-git"): None, (41496, "src-d/go-git"): None,
            (41495, "src-d/go-git"): None, (41500, "src-d/go-git"): None,
            (41499, "src-d/go-git"): None, (41498, "src-d/go-git"): None,
            (41497, "src-d/go-git"): None, (41490, "src-d/go-git"): None,
            (41488, "src-d/go-git"): None, (41487, "src-d/go-git"): None,
            (41486, "src-d/go-git"): None, (41494, "src-d/go-git"): None,
            (41493, "src-d/go-git"): None, (41492, "src-d/go-git"): None,
            (41491, "src-d/go-git"): None},
        "age": {
            (41475, "src-d/go-git"): pd.Timedelta("1 days 01:44:14"),
            (41474, "src-d/go-git"): pd.Timedelta("42 days 14:43:54"),
            (41473, "src-d/go-git"): pd.Timedelta("61 days 12:00:20"),
            (41472, "src-d/go-git"): pd.Timedelta("62 days 23:02:57"),
            (41471, "src-d/go-git"): pd.Timedelta("14 days 17:40:53"),
            (41470, "src-d/go-git"): pd.Timedelta("0 days 08:09:17"),
            (41469, "src-d/go-git"): pd.Timedelta("63 days 18:40:09"),
            (41468, "src-d/go-git"): pd.Timedelta("7 days 16:47:44"),
            (41467, "src-d/go-git"): pd.Timedelta("34 days 12:49:02"),
            (41485, "src-d/go-git"): pd.Timedelta("39 days 02:41:13"),
            (41484, "src-d/go-git"): pd.Timedelta("20 days 20:44:28"),
            (41483, "src-d/go-git"): pd.Timedelta("38 days 00:24:01"),
            (41482, "src-d/go-git"): pd.Timedelta("32 days 02:17:06"),
            (41481, "src-d/go-git"): pd.Timedelta("22 days 21:06:20"),
            (41480, "src-d/go-git"): pd.Timedelta("28 days 20:58:21"),
            (41479, "src-d/go-git"): pd.Timedelta("6 days 05:04:00"),
            (41478, "src-d/go-git"): pd.Timedelta("8 days 00:24:48"),
            (41477, "src-d/go-git"): pd.Timedelta("21 days 21:36:25"),
            (41476, "src-d/go-git"): pd.Timedelta("22 days 22:10:14"),
            (41517, "src-d/go-git"): pd.Timedelta("31 days 22:37:43"),
            (41519, "src-d/go-git"): pd.Timedelta("8 days 00:35:15"),
            (41518, "src-d/go-git"): pd.Timedelta("125 days 19:12:28"),
            (41514, "src-d/go-git"): pd.Timedelta("6 days 22:37:02"),
            (41513, "src-d/go-git"): pd.Timedelta("30 days 23:13:34"),
            (41516, "src-d/go-git"): pd.Timedelta("11 days 12:02:59"),
            (41515, "src-d/go-git"): pd.Timedelta("24 days 09:43:53"),
            (41512, "src-d/go-git"): pd.Timedelta("29 days 12:04:34"),
            (41511, "src-d/go-git"): pd.Timedelta("112 days 00:21:38"),
            (41510, "src-d/go-git"): pd.Timedelta("0 days 23:48:24"),
            (41509, "src-d/go-git"): pd.Timedelta("11 days 20:29:13"),
            (41508, "src-d/go-git"): pd.Timedelta("30 days 19:53:48"),
            (41506, "src-d/go-git"): pd.Timedelta("2 days 19:45:31"),
            (41505, "src-d/go-git"): pd.Timedelta("32 days 00:38:29"),
            (41503, "src-d/go-git"): pd.Timedelta("7 days 10:12:43"),
            (41502, "src-d/go-git"): pd.Timedelta("2 days 19:38:14"),
            (41501, "src-d/go-git"): pd.Timedelta("38 days 04:41:13"),
            (41507, "src-d/go-git"): pd.Timedelta("56 days 01:44:25"),
            (41496, "src-d/go-git"): pd.Timedelta("27 days 20:28:23"),
            (41495, "src-d/go-git"): pd.Timedelta("47 days 00:55:47"),
            (41500, "src-d/go-git"): pd.Timedelta("8 days 23:17:32"),
            (41499, "src-d/go-git"): pd.Timedelta("15 days 04:52:38"),
            (41498, "src-d/go-git"): pd.Timedelta("2 days 21:47:06"),
            (41497, "src-d/go-git"): pd.Timedelta("58 days 09:34:39"),
            (41490, "src-d/go-git"): pd.Timedelta("7 days 07:24:03"),
            (41488, "src-d/go-git"): pd.Timedelta("20 days 05:30:41"),
            (41487, "src-d/go-git"): pd.Timedelta("15 days 23:41:53"),
            (41486, "src-d/go-git"): pd.Timedelta("2 days 19:24:15"),
            (41494, "src-d/go-git"): pd.Timedelta("1 days 00:57:12"),
            (41493, "src-d/go-git"): pd.Timedelta("25 days 14:50:16"),
            (41492, "src-d/go-git"): pd.Timedelta("51 days 13:01:36"),
            (41491, "src-d/go-git"): pd.Timedelta("199 days 09:01:05")},
        "additions": {
            (41475, "src-d/go-git"): 2, (41474, "src-d/go-git"): 499,
            (41473, "src-d/go-git"): 11841, (41472, "src-d/go-git"): 633,
            (41471, "src-d/go-git"): 2922, (41470, "src-d/go-git"): 42,
            (41469, "src-d/go-git"): 1177, (41468, "src-d/go-git"): 114,
            (41467, "src-d/go-git"): 1971, (41485, "src-d/go-git"): 10461,
            (41484, "src-d/go-git"): 1886,
            (41483, "src-d/go-git"): 10316, (41482, "src-d/go-git"): 980,
            (41481, "src-d/go-git"): 396, (41480, "src-d/go-git"): 453,
            (41479, "src-d/go-git"): 380, (41478, "src-d/go-git"): 1160,
            (41477, "src-d/go-git"): 747, (41476, "src-d/go-git"): 2497,
            (41517, "src-d/go-git"): 625, (41519, "src-d/go-git"): 304,
            (41518, "src-d/go-git"): 10699, (41514, "src-d/go-git"): 1000,
            (41513, "src-d/go-git"): 1985, (41516, "src-d/go-git"): 6312,
            (41515, "src-d/go-git"): 2052, (41512, "src-d/go-git"): 3285,
            (41511, "src-d/go-git"): 30744, (41510, "src-d/go-git"): 323,
            (41509, "src-d/go-git"): 6473, (41508, "src-d/go-git"): 2734,
            (41506, "src-d/go-git"): 48, (41505, "src-d/go-git"): 14670,
            (41503, "src-d/go-git"): 2814, (41502, "src-d/go-git"): 37,
            (41501, "src-d/go-git"): 8942, (41507, "src-d/go-git"): 17069,
            (41496, "src-d/go-git"): 230, (41495, "src-d/go-git"): 10317,
            (41500, "src-d/go-git"): 16, (41499, "src-d/go-git"): 10,
            (41498, "src-d/go-git"): 114, (41497, "src-d/go-git"): 2399,
            (41490, "src-d/go-git"): 3327, (41488, "src-d/go-git"): 392,
            (41487, "src-d/go-git"): 620, (41486, "src-d/go-git"): 26,
            (41494, "src-d/go-git"): 68, (41493, "src-d/go-git"): 407,
            (41492, "src-d/go-git"): 10464, (41491, "src-d/go-git"): 1884},
        "deletions": {
            (41475, "src-d/go-git"): 2, (41474, "src-d/go-git"): 186,
            (41473, "src-d/go-git"): 1664, (41472, "src-d/go-git"): 80,
            (41471, "src-d/go-git"): 144, (41470, "src-d/go-git"): 10,
            (41469, "src-d/go-git"): 376, (41468, "src-d/go-git"): 38,
            (41467, "src-d/go-git"): 338, (41485, "src-d/go-git"): 2857,
            (41484, "src-d/go-git"): 283, (41483, "src-d/go-git"): 4930,
            (41482, "src-d/go-git"): 242, (41481, "src-d/go-git"): 108,
            (41480, "src-d/go-git"): 28, (41479, "src-d/go-git"): 26, (41478, "src-d/go-git"): 26,
            (41477, "src-d/go-git"): 107,
            (41476, "src-d/go-git"): 374, (41517, "src-d/go-git"): 89,
            (41519, "src-d/go-git"): 133, (41518, "src-d/go-git"): 3354,
            (41514, "src-d/go-git"): 330, (41513, "src-d/go-git"): 515,
            (41516, "src-d/go-git"): 2475, (41515, "src-d/go-git"): 830,
            (41512, "src-d/go-git"): 954, (41511, "src-d/go-git"): 12512,
            (41510, "src-d/go-git"): 146, (41509, "src-d/go-git"): 3562,
            (41508, "src-d/go-git"): 556, (41506, "src-d/go-git"): 1185,
            (41505, "src-d/go-git"): 6763, (41503, "src-d/go-git"): 2858,
            (41502, "src-d/go-git"): 3, (41501, "src-d/go-git"): 1551,
            (41507, "src-d/go-git"): 13498, (41496, "src-d/go-git"): 13,
            (41495, "src-d/go-git"): 978, (41500, "src-d/go-git"): 14, (41499, "src-d/go-git"): 12,
            (41498, "src-d/go-git"): 14,
            (41497, "src-d/go-git"): 616, (41490, "src-d/go-git"): 883,
            (41488, "src-d/go-git"): 163, (41487, "src-d/go-git"): 54, (41486, "src-d/go-git"): 6,
            (41494, "src-d/go-git"): 2,
            (41493, "src-d/go-git"): 101, (41492, "src-d/go-git"): 3402,
            (41491, "src-d/go-git"): 287},
        "commits_count": {
            (41475, "src-d/go-git"): 2, (41474, "src-d/go-git"): 20, (41473, "src-d/go-git"): 88,
            (41472, "src-d/go-git"): 32, (41471, "src-d/go-git"): 19, (41470, "src-d/go-git"): 2,
            (41469, "src-d/go-git"): 13, (41468, "src-d/go-git"): 2, (41467, "src-d/go-git"): 35,
            (41485, "src-d/go-git"): 62, (41484, "src-d/go-git"): 39, (41483, "src-d/go-git"): 60,
            (41482, "src-d/go-git"): 14, (41481, "src-d/go-git"): 14, (41480, "src-d/go-git"): 6,
            (41479, "src-d/go-git"): 10, (41478, "src-d/go-git"): 6, (41477, "src-d/go-git"): 17,
            (41476, "src-d/go-git"): 21, (41517, "src-d/go-git"): 21, (41519, "src-d/go-git"): 13,
            (41518, "src-d/go-git"): 181, (41514, "src-d/go-git"): 11, (41513, "src-d/go-git"): 48,
            (41516, "src-d/go-git"): 62, (41515, "src-d/go-git"): 42, (41512, "src-d/go-git"): 46,
            (41511, "src-d/go-git"): 195, (41510, "src-d/go-git"): 7, (41509, "src-d/go-git"): 34,
            (41508, "src-d/go-git"): 14, (41506, "src-d/go-git"): 2, (41505, "src-d/go-git"): 58,
            (41503, "src-d/go-git"): 6, (41502, "src-d/go-git"): 3, (41501, "src-d/go-git"): 23,
            (41507, "src-d/go-git"): 110, (41496, "src-d/go-git"): 2, (41495, "src-d/go-git"): 8,
            (41500, "src-d/go-git"): 2, (41499, "src-d/go-git"): 2, (41498, "src-d/go-git"): 4,
            (41497, "src-d/go-git"): 12,
            (41490, "src-d/go-git"): 25, (41488, "src-d/go-git"): 30, (41487, "src-d/go-git"): 9,
            (41486, "src-d/go-git"): 2, (41494, "src-d/go-git"): 2, (41493, "src-d/go-git"): 4,
            (41492, "src-d/go-git"): 47,
            (41491, "src-d/go-git"): 21},
        "repository_node_id": {
            (41475, "src-d/go-git"): 40550, (41474, "src-d/go-git"): 40550,
            (41473, "src-d/go-git"): 40550, (41472, "src-d/go-git"): 40550,
            (41471, "src-d/go-git"): 40550, (41470, "src-d/go-git"): 40550,
            (41469, "src-d/go-git"): 40550, (41468, "src-d/go-git"): 40550,
            (41467, "src-d/go-git"): 40550, (41485, "src-d/go-git"): 40550,
            (41484, "src-d/go-git"): 40550, (41483, "src-d/go-git"): 40550,
            (41482, "src-d/go-git"): 40550, (41481, "src-d/go-git"): 40550,
            (41480, "src-d/go-git"): 40550, (41479, "src-d/go-git"): 40550,
            (41478, "src-d/go-git"): 40550, (41477, "src-d/go-git"): 40550,
            (41476, "src-d/go-git"): 40550, (41517, "src-d/go-git"): 40550,
            (41519, "src-d/go-git"): 40550, (41518, "src-d/go-git"): 40550,
            (41514, "src-d/go-git"): 40550, (41513, "src-d/go-git"): 40550,
            (41516, "src-d/go-git"): 40550, (41515, "src-d/go-git"): 40550,
            (41512, "src-d/go-git"): 40550, (41511, "src-d/go-git"): 40550,
            (41510, "src-d/go-git"): 40550, (41509, "src-d/go-git"): 40550,
            (41508, "src-d/go-git"): 40550, (41506, "src-d/go-git"): 40550,
            (41505, "src-d/go-git"): 40550, (41503, "src-d/go-git"): 40550,
            (41502, "src-d/go-git"): 40550, (41501, "src-d/go-git"): 40550,
            (41507, "src-d/go-git"): 40550, (41496, "src-d/go-git"): 40550,
            (41495, "src-d/go-git"): 40550, (41500, "src-d/go-git"): 40550,
            (41499, "src-d/go-git"): 40550, (41498, "src-d/go-git"): 40550,
            (41497, "src-d/go-git"): 40550, (41490, "src-d/go-git"): 40550,
            (41488, "src-d/go-git"): 40550, (41487, "src-d/go-git"): 40550,
            (41486, "src-d/go-git"): 40550, (41494, "src-d/go-git"): 40550,
            (41493, "src-d/go-git"): 40550, (41492, "src-d/go-git"): 40550,
            (41491, "src-d/go-git"): 40550},
        "author_node_id": {
            (41475, "src-d/go-git"): 39789, (41474, "src-d/go-git"): 39789,
            (41473, "src-d/go-git"): 39789, (41472, "src-d/go-git"): 39789,
            (41471, "src-d/go-git"): 39789,
            (41470, "src-d/go-git"): 39789, (41469, "src-d/go-git"): 39789,
            (41468, "src-d/go-git"): 39789, (41467, "src-d/go-git"): 39789,
            (41485, "src-d/go-git"): 39789,
            (41484, "src-d/go-git"): 39789, (41483, "src-d/go-git"): 39789,
            (41482, "src-d/go-git"): 39789, (41481, "src-d/go-git"): 39789,
            (41480, "src-d/go-git"): 39789,
            (41479, "src-d/go-git"): 39789, (41478, "src-d/go-git"): 39789,
            (41477, "src-d/go-git"): 39789, (41476, "src-d/go-git"): 39789,
            (41517, "src-d/go-git"): 39789,
            (41519, "src-d/go-git"): 39789, (41518, "src-d/go-git"): 39789,
            (41514, "src-d/go-git"): 39789, (41513, "src-d/go-git"): 39789,
            (41516, "src-d/go-git"): 39789,
            (41515, "src-d/go-git"): 39789, (41512, "src-d/go-git"): 39789,
            (41511, "src-d/go-git"): 40070, (41510, "src-d/go-git"): 39789,
            (41509, "src-d/go-git"): 39789,
            (41508, "src-d/go-git"): 40070, (41506, "src-d/go-git"): 40070,
            (41505, "src-d/go-git"): 39789, (41503, "src-d/go-git"): 39789,
            (41502, "src-d/go-git"): 40070,
            (41501, "src-d/go-git"): 40070, (41507, "src-d/go-git"): 39789,
            (41496, "src-d/go-git"): 39789, (41495, "src-d/go-git"): 40418,
            (41500, "src-d/go-git"): 39789,
            (41499, "src-d/go-git"): 39789, (41498, "src-d/go-git"): 39789,
            (41497, "src-d/go-git"): 39789, (41490, "src-d/go-git"): 39789,
            (41488, "src-d/go-git"): 39789,
            (41487, "src-d/go-git"): 39789, (41486, "src-d/go-git"): 39789,
            (41494, "src-d/go-git"): 39789, (41493, "src-d/go-git"): 39789,
            (41492, "src-d/go-git"): 39789,
            (41491, "src-d/go-git"): 39789},
        "name": {
            (41475, "src-d/go-git"): "v4.13.1", (41474, "src-d/go-git"): "v4.13.0",
            (41473, "src-d/go-git"): "v4.12.0", (41472, "src-d/go-git"): "v4.11.0",
            (41471, "src-d/go-git"): "v4.10.0", (41470, "src-d/go-git"): "v4.9.1",
            (41469, "src-d/go-git"): "v4.9.0", (41468, "src-d/go-git"): "v4.8.1",
            (41467, "src-d/go-git"): "v4.8.0", (41485, "src-d/go-git"): "v4.7.1",
            (41484, "src-d/go-git"): "v4.7.0", (41483, "src-d/go-git"): "v4.6.0",
            (41482, "src-d/go-git"): "v4.5.0",
            (41481, "src-d/go-git"): "v4.4.1", (41480, "src-d/go-git"): "v4.4.0",
            (41479, "src-d/go-git"): "v4.3.1", (41478, "src-d/go-git"): "v4.3.0",
            (41477, "src-d/go-git"): "v4.2.1", (41476, "src-d/go-git"): "v4.2.0",
            (41517, "src-d/go-git"): "v4.1.1", (41519, "src-d/go-git"): "v4.1.0",
            (41518, "src-d/go-git"): "v4.0.0", (41514, "src-d/go-git"): "v4.0.0-rc15",
            (41513, "src-d/go-git"): "v4.0.0-rc14", (41516, "src-d/go-git"): "v4.0.0-rc13",
            (41515, "src-d/go-git"): "v4.0.0-rc12", (41512, "src-d/go-git"): "v4.0.0-rc11",
            (41511, "src-d/go-git"): "v4.0.0-rc10",
            (41510, "src-d/go-git"): "v4.0.0-rc9", (41509, "src-d/go-git"): "v4.0.0-rc8",
            (41508, "src-d/go-git"): "v4.0.0-rc7",
            (41506, "src-d/go-git"): "v4.0.0-rc6", (41505, "src-d/go-git"): "v4.0.0-rc5",
            (41503, "src-d/go-git"): "v4.0.0-rc4",
            (41502, "src-d/go-git"): "v4.0.0-rc3", (41501, "src-d/go-git"): "v4.0.0-rc2",
            (41507, "src-d/go-git"): "v4.0.0-rc1", (41496, "src-d/go-git"): "v3.1.1",
            (41495, "src-d/go-git"): "v3.1.0", (41500, "src-d/go-git"): "v3.0.4",
            (41499, "src-d/go-git"): "v3.0.3", (41498, "src-d/go-git"): "v3.0.2",
            (41497, "src-d/go-git"): "v3.0.1",
            (41490, "src-d/go-git"): "v3.0.0-alpha", (41488, "src-d/go-git"): "v2.2.0",
            (41487, "src-d/go-git"): "v2.1.3", (41486, "src-d/go-git"): "v2.1.2 hotfix",
            (41494, "src-d/go-git"): "v2.1.1 hotfix", (41493, "src-d/go-git"): "v2.1.0",
            (41492, "src-d/go-git"): "v2.0.0", (41491, "src-d/go-git"): "v1.0.0"},
        "published_at": {
            (41475, "src-d/go-git"): pd.Timestamp("2019-08-01 15:25:42+0000", tz="UTC"),
            (41474, "src-d/go-git"): pd.Timestamp("2019-07-31 13:41:28+0000", tz="UTC"),
            (41473, "src-d/go-git"): pd.Timestamp("2019-06-18 22:57:34+0000", tz="UTC"),
            (41472, "src-d/go-git"): pd.Timestamp("2019-04-18 10:57:14+0000", tz="UTC"),
            (41471, "src-d/go-git"): pd.Timestamp("2019-02-14 11:54:17+0000", tz="UTC"),
            (41470, "src-d/go-git"): pd.Timestamp("2019-01-30 18:13:24+0000", tz="UTC"),
            (41469, "src-d/go-git"): pd.Timestamp("2019-01-30 10:04:07+0000", tz="UTC"),
            (41468, "src-d/go-git"): pd.Timestamp("2018-11-27 15:23:58+0000", tz="UTC"),
            (41467, "src-d/go-git"): pd.Timestamp("2018-11-19 22:36:14+0000", tz="UTC"),
            (41485, "src-d/go-git"): pd.Timestamp("2018-10-16 09:47:12+0000", tz="UTC"),
            (41484, "src-d/go-git"): pd.Timestamp("2018-09-07 07:05:59+0000", tz="UTC"),
            (41483, "src-d/go-git"): pd.Timestamp("2018-08-17 10:21:31+0000", tz="UTC"),
            (41482, "src-d/go-git"): pd.Timestamp("2018-07-10 09:57:30+0000", tz="UTC"),
            (41481, "src-d/go-git"): pd.Timestamp("2018-06-08 07:40:24+0000", tz="UTC"),
            (41480, "src-d/go-git"): pd.Timestamp("2018-05-16 10:34:04+0000", tz="UTC"),
            (41479, "src-d/go-git"): pd.Timestamp("2018-04-17 13:35:43+0000", tz="UTC"),
            (41478, "src-d/go-git"): pd.Timestamp("2018-04-11 08:31:43+0000", tz="UTC"),
            (41477, "src-d/go-git"): pd.Timestamp("2018-04-03 08:06:55+0000", tz="UTC"),
            (41476, "src-d/go-git"): pd.Timestamp("2018-03-12 10:30:30+0000", tz="UTC"),
            (41517, "src-d/go-git"): pd.Timestamp("2018-02-17 12:20:16+0000", tz="UTC"),
            (41519, "src-d/go-git"): pd.Timestamp("2018-01-16 13:42:33+0000", tz="UTC"),
            (41518, "src-d/go-git"): pd.Timestamp("2018-01-08 13:07:18+0000", tz="UTC"),
            (41514, "src-d/go-git"): pd.Timestamp("2017-09-04 17:54:50+0000", tz="UTC"),
            (41513, "src-d/go-git"): pd.Timestamp("2017-08-28 19:17:48+0000", tz="UTC"),
            (41516, "src-d/go-git"): pd.Timestamp("2017-07-28 20:04:14+0000", tz="UTC"),
            (41515, "src-d/go-git"): pd.Timestamp("2017-07-17 08:01:15+0000", tz="UTC"),
            (41512, "src-d/go-git"): pd.Timestamp("2017-06-22 22:17:22+0000", tz="UTC"),
            (41511, "src-d/go-git"): pd.Timestamp("2017-05-24 10:12:48+0000", tz="UTC"),
            (41510, "src-d/go-git"): pd.Timestamp("2017-02-01 09:51:10+0000", tz="UTC"),
            (41509, "src-d/go-git"): pd.Timestamp("2017-01-31 10:02:46+0000", tz="UTC"),
            (41508, "src-d/go-git"): pd.Timestamp("2017-01-19 13:33:33+0000", tz="UTC"),
            (41506, "src-d/go-git"): pd.Timestamp("2016-12-19 17:39:45+0000", tz="UTC"),
            (41505, "src-d/go-git"): pd.Timestamp("2016-12-16 21:54:14+0000", tz="UTC"),
            (41503, "src-d/go-git"): pd.Timestamp("2016-11-14 21:15:45+0000", tz="UTC"),
            (41502, "src-d/go-git"): pd.Timestamp("2016-11-07 11:03:02+0000", tz="UTC"),
            (41501, "src-d/go-git"): pd.Timestamp("2016-11-04 15:24:48+0000", tz="UTC"),
            (41507, "src-d/go-git"): pd.Timestamp("2016-09-27 10:43:35+0000", tz="UTC"),
            (41496, "src-d/go-git"): pd.Timestamp("2016-08-02 08:59:10+0000", tz="UTC"),
            (41495, "src-d/go-git"): pd.Timestamp("2016-07-05 12:30:47+0000", tz="UTC"),
            (41500, "src-d/go-git"): pd.Timestamp("2016-05-19 11:35:00+0000", tz="UTC"),
            (41499, "src-d/go-git"): pd.Timestamp("2016-05-10 12:17:28+0000", tz="UTC"),
            (41498, "src-d/go-git"): pd.Timestamp("2016-04-25 07:24:50+0000", tz="UTC"),
            (41497, "src-d/go-git"): pd.Timestamp("2016-04-22 09:37:44+0000", tz="UTC"),
            (41490, "src-d/go-git"): pd.Timestamp("2016-02-24 00:03:05+0000", tz="UTC"),
            (41488, "src-d/go-git"): pd.Timestamp("2016-02-16 16:39:02+0000", tz="UTC"),
            (41487, "src-d/go-git"): pd.Timestamp("2016-01-27 11:08:21+0000", tz="UTC"),
            (41486, "src-d/go-git"): pd.Timestamp("2016-01-11 11:26:28+0000", tz="UTC"),
            (41494, "src-d/go-git"): pd.Timestamp("2016-01-08 16:02:13+0000", tz="UTC"),
            (41493, "src-d/go-git"): pd.Timestamp("2016-01-07 15:05:01+0000", tz="UTC"),
            (41492, "src-d/go-git"): pd.Timestamp("2015-12-13 00:14:45+0000", tz="UTC"),
            (41491, "src-d/go-git"): pd.Timestamp("2015-10-22 11:13:09+0000", tz="UTC")},
        "tag": {
            (41475, "src-d/go-git"): "v4.13.1", (41474, "src-d/go-git"): "v4.13.0",
            (41473, "src-d/go-git"): "v4.12.0", (41472, "src-d/go-git"): "v4.11.0",
            (41471, "src-d/go-git"): "v4.10.0", (41470, "src-d/go-git"): "v4.9.1",
            (41469, "src-d/go-git"): "v4.9.0", (41468, "src-d/go-git"): "v4.8.1",
            (41467, "src-d/go-git"): "v4.8.0", (41485, "src-d/go-git"): "v4.7.1",
            (41484, "src-d/go-git"): "v4.7.0", (41483, "src-d/go-git"): "v4.6.0",
            (41482, "src-d/go-git"): "v4.5.0", (41481, "src-d/go-git"): "v4.4.1",
            (41480, "src-d/go-git"): "v4.4.0", (41479, "src-d/go-git"): "v4.3.1",
            (41478, "src-d/go-git"): "v4.3.0", (41477, "src-d/go-git"): "v4.2.1",
            (41476, "src-d/go-git"): "v4.2.0", (41517, "src-d/go-git"): "v4.1.1",
            (41519, "src-d/go-git"): "v4.1.0", (41518, "src-d/go-git"): "v4.0.0",
            (41514, "src-d/go-git"): "v4.0.0-rc15", (41513, "src-d/go-git"): "v4.0.0-rc14",
            (41516, "src-d/go-git"): "v4.0.0-rc13", (41515, "src-d/go-git"): "v4.0.0-rc12",
            (41512, "src-d/go-git"): "v4.0.0-rc11",
            (41511, "src-d/go-git"): "v4.0.0-rc10", (41510, "src-d/go-git"): "v4.0.0-rc9",
            (41509, "src-d/go-git"): "v4.0.0-rc8",
            (41508, "src-d/go-git"): "v4.0.0-rc7", (41506, "src-d/go-git"): "v4.0.0-rc6",
            (41505, "src-d/go-git"): "v4.0.0-rc5", (41503, "src-d/go-git"): "v4.0.0-rc4",
            (41502, "src-d/go-git"): "v4.0.0-rc3", (41501, "src-d/go-git"): "v4.0.0-rc2",
            (41507, "src-d/go-git"): "v4.0.0-rc1", (41496, "src-d/go-git"): "v3.1.1",
            (41495, "src-d/go-git"): "v3.1.0", (41500, "src-d/go-git"): "v3.0.4",
            (41499, "src-d/go-git"): "v3.0.3", (41498, "src-d/go-git"): "v3.0.2",
            (41497, "src-d/go-git"): "v3.0.1", (41490, "src-d/go-git"): "v3.0.0",
            (41488, "src-d/go-git"): "v2.2.0", (41487, "src-d/go-git"): "v2.1.3",
            (41486, "src-d/go-git"): "v2.1.2", (41494, "src-d/go-git"): "v2.1.1",
            (41493, "src-d/go-git"): "v2.1.0", (41492, "src-d/go-git"): "v2.0.0",
            (41491, "src-d/go-git"): "v1.0.0"},
        "url": {
            (41475, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.13.1",
            (41474, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.13.0",
            (41473, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.12.0",
            (41472, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.11.0",
            (41471, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.10.0",
            (41470, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.9.1",
            (41469, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.9.0",
            (41468, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.8.1",
            (41467, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.8.0",
            (41485, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.7.1",
            (41484, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.7.0",
            (41483, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.6.0",
            (41482, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.5.0",
            (41481, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.4.1",
            (41480, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.4.0",
            (41479, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.3.1",
            (41478, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.3.0",
            (41477, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.2.1",
            (41476, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.2.0",
            (41517, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.1.1",
            (41519, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.1.0",
            (41518, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0",
            (41514, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc15",
            (41513, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc14",
            (41516, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc13",
            (41515, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc12",
            (41512, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc11",
            (41511, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc10",
            (41510, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc9",
            (41509, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc8",
            (41508, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc7",
            (41506, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc6",
            (41505, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc5",
            (41503, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc4",
            (41502, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc3",
            (41501, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc2",
            (41507, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc1",
            (41496, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v3.1.1",
            (41495, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v3.1.0",
            (41500, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v3.0.4",
            (41499, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v3.0.3",
            (41498, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v3.0.2",
            (41497, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v3.0.1",
            (41490, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v3.0.0",
            (41488, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v2.2.0",
            (41487, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v2.1.3",
            (41486, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v2.1.2",
            (41494, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v2.1.1",
            (41493, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v2.1.0",
            (41492, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v2.0.0",
            (41491, "src-d/go-git"): "https://github.com/src-d/go-git/releases/tag/v1.0.0"},
        "sha": {
            (41475, "src-d/go-git"): "0d1a009cbb604db18be960db5f1525b99a55d727",
            (41474, "src-d/go-git"): "6241d0e70427cb0db4ca00182717af88f638268c",
            (41473, "src-d/go-git"): "f9a30199e7083bdda8adad3a4fa2ec42d25c1fdb",
            (41472, "src-d/go-git"): "aa6f288c256ff8baf8a7745546a9752323dc0d89",
            (41471, "src-d/go-git"): "db6c41c156481962abf9a55a324858674c25ab08",
            (41470, "src-d/go-git"): "a1f6ef44dfed1253ef7f3bc049f66b15f8fc2ab2",
            (41469, "src-d/go-git"): "434611b74cb54538088c6aeed4ed27d3044064fa",
            (41468, "src-d/go-git"): "3dbfb89e0f5bce0008724e547b999fe3af9f60db",
            (41467, "src-d/go-git"): "f62cd8e3495579a8323455fa0c4e6c44bb0d5e09",
            (41485, "src-d/go-git"): "cd64b4d630b6c2d2b3d72e9615e14f9d58bb5787",
            (41484, "src-d/go-git"): "d3cec13ac0b195bfb897ed038a08b5130ab9969e",
            (41483, "src-d/go-git"): "7b6c1266556f59ac436fada3fa6106d4a84f9b56",
            (41482, "src-d/go-git"): "3bd5e82b2512d85becae9677fa06b5a973fd4cfb",
            (41481, "src-d/go-git"): "b23570073eaee3489e5e3d666f22ba5cbeb53243",
            (41480, "src-d/go-git"): "57570e84f8c5739f0f4a59387493e590e709dde9",
            (41479, "src-d/go-git"): "b30763cb64afa91c016b23e905af0a378eb1b76d",
            (41478, "src-d/go-git"): "0db54e829f81a28f71c22d54c03daba5ec144c8d",
            (41477, "src-d/go-git"): "247cf690745dfd67ccd9f0c07878e6dd85e6c9ed",
            (41476, "src-d/go-git"): "1d28459504251497e0ce6132a0fadd5eb44ffd22",
            (41517, "src-d/go-git"): "886dc83f3ed518a78772055497bcc7d7621b468e",
            (41519, "src-d/go-git"): "e9247ce9c5ce12126f646ca3ddf0066e4829bd14",
            (41518, "src-d/go-git"): "bf3b1f1fb9e0a04d0f87511a7ded2562b48a19d8",
            (41514, "src-d/go-git"): "f9879dd043f84936a1f8acb8a53b74332a7ae135",
            (41513, "src-d/go-git"): "7aa9d15d395282144f31a09c0fac230da3f65360",
            (41516, "src-d/go-git"): "8ddbecf782c2e340fd85bb4ba4d00dc73d749f87",
            (41515, "src-d/go-git"): "d3c7400c39f86a4c59340c7a9cda8497186e00fc",
            (41512, "src-d/go-git"): "ad02bf020460c210660db4fffda7f926b6aae95a",
            (41511, "src-d/go-git"): "7e249dfcf28765939bde8f38784b3274b522f880",
            (41510, "src-d/go-git"): "a9920b123ba1f6819a8c03209582d4d28e9fd831",
            (41509, "src-d/go-git"): "f84e3bbfe59f5438c90000e6a89b41ec8eab51fb",
            (41508, "src-d/go-git"): "441713897ef5604e8105379f45ebb982ab2c9a75",
            (41506, "src-d/go-git"): "e42e10f112ec93fbca3d972dffa9566b94a0f6f8",
            (41505, "src-d/go-git"): "c9353b2bd7c1cbdf8f78dad6deac64ed2f2ed9eb",
            (41503, "src-d/go-git"): "eb89d2dd9a36440d58aea224c055b364e49785f7",
            (41502, "src-d/go-git"): "f6ed7424cbf33c7013332d7e95b4262a4bc4a523",
            (41501, "src-d/go-git"): "743989abd8c1277dff78e56c2583a9f6dff796ff",
            (41507, "src-d/go-git"): "8cd772a53e8ecd2687b739eea110fa9b179f1e0f",
            (41496, "src-d/go-git"): "5413c7aeadb7cb18a6d51dae0bc313f2e129a337",
            (41495, "src-d/go-git"): "5e73f01cb2e027a8f02801635b79d3a9bc866914",
            (41500, "src-d/go-git"): "08f9e7015aad2ca768638b446fb8632f11601899",
            (41499, "src-d/go-git"): "1cd347ec8970388f83745c9a530ea2bcd705c6d9",
            (41498, "src-d/go-git"): "36d14454b32eca89ac43d2934c50f3a1ae2e1d20",
            (41497, "src-d/go-git"): "b08327bfaf27171dddc5516c63e5646c40f0b004",
            (41490, "src-d/go-git"): "07ca1ac7f3058ea6d3274a01973541fb84782f5e",
            (41488, "src-d/go-git"): "1931dfbf38508e790e9f129873bc073aacc6a50f",
            (41487, "src-d/go-git"): "35ee4d749be21691b78a7465361ad47179fe2eff",
            (41486, "src-d/go-git"): "37cc5cf842c3c0fb989bcf7525cc8f826d96b295",
            (41494, "src-d/go-git"): "cebec78608e7913b8c843390237fd609069022ae",
            (41493, "src-d/go-git"): "da5ab9de3e4c1bffa533108f46c5adc30929f7c2",
            (41492, "src-d/go-git"): "f821e1340752dce95f73375dc9a13dcd58d58f82",
            (41491, "src-d/go-git"): "6f43e8933ba3c04072d5d104acc6118aac3e52ee"},
        "commit_id": {
            (41475, "src-d/go-git"): 2755244, (41474, "src-d/go-git"): 2756276,
            (41473, "src-d/go-git"): 2758155, (41472, "src-d/go-git"): 2757229,
            (41471, "src-d/go-git"): 2757785, (41470, "src-d/go-git"): 2757112,
            (41469, "src-d/go-git"): 2755886, (41468, "src-d/go-git"): 2755046,
            (41467, "src-d/go-git"): 2758121, (41485, "src-d/go-git"): 2757058,
            (41484, "src-d/go-git"): 2757690, (41483, "src-d/go-git"): 2755789,
            (41482, "src-d/go-git"): 2755028, (41481, "src-d/go-git"): 2757357,
            (41480, "src-d/go-git"): 2756138, (41479, "src-d/go-git"): 2757371,
            (41478, "src-d/go-git"): 2755246, (41477, "src-d/go-git"): 2755522,
            (41476, "src-d/go-git"): 2755428, (41517, "src-d/go-git"): 2756702,
            (41519, "src-d/go-git"): 2757970, (41518, "src-d/go-git"): 2757510,
            (41514, "src-d/go-git"): 2758150, (41513, "src-d/go-git"): 2755784,
            (41516, "src-d/go-git"): 2756777, (41515, "src-d/go-git"): 2757689,
            (41512, "src-d/go-git"): 2757276, (41511, "src-d/go-git"): 2755834,
            (41510, "src-d/go-git"): 2757219, (41509, "src-d/go-git"): 2758137,
            (41508, "src-d/go-git"): 2755902, (41506, "src-d/go-git"): 2757904,
            (41505, "src-d/go-git"): 2757629, (41503, "src-d/go-git"): 2757988,
            (41502, "src-d/go-git"): 2758123, (41501, "src-d/go-git"): 2756516,
            (41507, "src-d/go-git"): 2756766, (41496, "src-d/go-git"): 2756112,
            (41495, "src-d/go-git"): 2756224, (41500, "src-d/go-git"): 2755176,
            (41499, "src-d/go-git"): 2755419, (41498, "src-d/go-git"): 2755730,
            (41497, "src-d/go-git"): 2757320, (41490, "src-d/go-git"): 2755165,
            (41488, "src-d/go-git"): 2755383, (41487, "src-d/go-git"): 2755718,
            (41486, "src-d/go-git"): 2755745, (41494, "src-d/go-git"): 2757079,
            (41493, "src-d/go-git"): 2757770, (41492, "src-d/go-git"): 2758144,
            (41491, "src-d/go-git"): 2756452},
        "matched_by": {
            (41475, "src-d/go-git"): 1, (41474, "src-d/go-git"): 1, (41473, "src-d/go-git"): 1,
            (41472, "src-d/go-git"): 1, (41471, "src-d/go-git"): 1, (41470, "src-d/go-git"): 1,
            (41469, "src-d/go-git"): 1,
            (41468, "src-d/go-git"): 1, (41467, "src-d/go-git"): 1, (41485, "src-d/go-git"): 1,
            (41484, "src-d/go-git"): 1, (41483, "src-d/go-git"): 1, (41482, "src-d/go-git"): 1,
            (41481, "src-d/go-git"): 1,
            (41480, "src-d/go-git"): 1, (41479, "src-d/go-git"): 1, (41478, "src-d/go-git"): 1,
            (41477, "src-d/go-git"): 1, (41476, "src-d/go-git"): 1, (41517, "src-d/go-git"): 1,
            (41519, "src-d/go-git"): 1,
            (41518, "src-d/go-git"): 1, (41514, "src-d/go-git"): 1, (41513, "src-d/go-git"): 1,
            (41516, "src-d/go-git"): 1, (41515, "src-d/go-git"): 1, (41512, "src-d/go-git"): 1,
            (41511, "src-d/go-git"): 1,
            (41510, "src-d/go-git"): 1, (41509, "src-d/go-git"): 1, (41508, "src-d/go-git"): 1,
            (41506, "src-d/go-git"): 1, (41505, "src-d/go-git"): 1, (41503, "src-d/go-git"): 1,
            (41502, "src-d/go-git"): 1,
            (41501, "src-d/go-git"): 1, (41507, "src-d/go-git"): 1, (41496, "src-d/go-git"): 1,
            (41495, "src-d/go-git"): 1, (41500, "src-d/go-git"): 1, (41499, "src-d/go-git"): 1,
            (41498, "src-d/go-git"): 1,
            (41497, "src-d/go-git"): 1, (41490, "src-d/go-git"): 1, (41488, "src-d/go-git"): 1,
            (41487, "src-d/go-git"): 1, (41486, "src-d/go-git"): 1, (41494, "src-d/go-git"): 1,
            (41493, "src-d/go-git"): 1,
            (41492, "src-d/go-git"): 1, (41491, "src-d/go-git"): 1},
        "author": {
            (41475, "src-d/go-git"): "mcuadros", (41474, "src-d/go-git"): "mcuadros",
            (41473, "src-d/go-git"): "mcuadros", (41472, "src-d/go-git"): "mcuadros",
            (41471, "src-d/go-git"): "mcuadros", (41470, "src-d/go-git"): "mcuadros",
            (41469, "src-d/go-git"): "mcuadros", (41468, "src-d/go-git"): "mcuadros",
            (41467, "src-d/go-git"): "mcuadros", (41485, "src-d/go-git"): "mcuadros",
            (41484, "src-d/go-git"): "mcuadros", (41483, "src-d/go-git"): "mcuadros",
            (41482, "src-d/go-git"): "mcuadros", (41481, "src-d/go-git"): "mcuadros",
            (41480, "src-d/go-git"): "mcuadros", (41479, "src-d/go-git"): "mcuadros",
            (41478, "src-d/go-git"): "mcuadros", (41477, "src-d/go-git"): "mcuadros",
            (41476, "src-d/go-git"): "mcuadros", (41517, "src-d/go-git"): "mcuadros",
            (41519, "src-d/go-git"): "mcuadros", (41518, "src-d/go-git"): "mcuadros",
            (41514, "src-d/go-git"): "mcuadros", (41513, "src-d/go-git"): "mcuadros",
            (41516, "src-d/go-git"): "mcuadros", (41515, "src-d/go-git"): "mcuadros",
            (41512, "src-d/go-git"): "mcuadros", (41511, "src-d/go-git"): "smola",
            (41510, "src-d/go-git"): "mcuadros", (41509, "src-d/go-git"): "mcuadros",
            (41508, "src-d/go-git"): "smola", (41506, "src-d/go-git"): "smola",
            (41505, "src-d/go-git"): "mcuadros", (41503, "src-d/go-git"): "mcuadros",
            (41502, "src-d/go-git"): "smola", (41501, "src-d/go-git"): "smola",
            (41507, "src-d/go-git"): "mcuadros", (41496, "src-d/go-git"): "mcuadros",
            (41495, "src-d/go-git"): "alcortesm", (41500, "src-d/go-git"): "mcuadros",
            (41499, "src-d/go-git"): "mcuadros", (41498, "src-d/go-git"): "mcuadros",
            (41497, "src-d/go-git"): "mcuadros", (41490, "src-d/go-git"): "mcuadros",
            (41488, "src-d/go-git"): "mcuadros", (41487, "src-d/go-git"): "mcuadros",
            (41486, "src-d/go-git"): "mcuadros", (41494, "src-d/go-git"): "mcuadros",
            (41493, "src-d/go-git"): "mcuadros", (41492, "src-d/go-git"): "mcuadros",
            (41491, "src-d/go-git"): "mcuadros"},
        "deployments": {
            (41467, "src-d/go-git"): None, (41468, "src-d/go-git"): None,
            (41469, "src-d/go-git"): None, (41470, "src-d/go-git"): None,
            (41471, "src-d/go-git"): None,
            (41472, "src-d/go-git"): None, (41473, "src-d/go-git"): None,
            (41474, "src-d/go-git"): None, (41475, "src-d/go-git"): None,
            (41476, "src-d/go-git"): None,
            (41477, "src-d/go-git"): None, (41478, "src-d/go-git"): None,
            (41479, "src-d/go-git"): None, (41480, "src-d/go-git"): None,
            (41481, "src-d/go-git"): None,
            (41482, "src-d/go-git"): None, (41483, "src-d/go-git"): None,
            (41484, "src-d/go-git"): None, (41485, "src-d/go-git"): None,
            (41486, "src-d/go-git"): None,
            (41487, "src-d/go-git"): None, (41488, "src-d/go-git"): None,
            (41490, "src-d/go-git"): None, (41491, "src-d/go-git"): None,
            (41492, "src-d/go-git"): None,
            (41493, "src-d/go-git"): None, (41494, "src-d/go-git"): None,
            (41495, "src-d/go-git"): None, (41496, "src-d/go-git"): None,
            (41497, "src-d/go-git"): None,
            (41498, "src-d/go-git"): None, (41499, "src-d/go-git"): None,
            (41500, "src-d/go-git"): None, (41501, "src-d/go-git"): None,
            (41502, "src-d/go-git"): None,
            (41503, "src-d/go-git"): None, (41505, "src-d/go-git"): None,
            (41506, "src-d/go-git"): None, (41507, "src-d/go-git"): None,
            (41508, "src-d/go-git"): None,
            (41509, "src-d/go-git"): None, (41510, "src-d/go-git"): None,
            (41511, "src-d/go-git"): None, (41512, "src-d/go-git"): None,
            (41513, "src-d/go-git"): None,
            (41514, "src-d/go-git"): None, (41515, "src-d/go-git"): None,
            (41516, "src-d/go-git"): None, (41517, "src-d/go-git"): None,
            (41518, "src-d/go-git"): None,
            (41519, "src-d/go-git"): None},
    }


@with_defer
async def test_mine_deployments_precomputed_dummy(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps1, people1 = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    deps2, people2 = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, None)
    assert len(deps1) == len(deps2) == 1
    assert deps1.index.tolist() == deps2.index.tolist()
    assert (rel1 := deps1["releases"].iloc[0]).columns.tolist() == \
           (rel2 := deps2["releases"].iloc[0]).columns.tolist()
    assert len(rel1) == len(rel2)
    assert (rel1.index == rel2.index).all()
    del deps1["releases"]
    del deps2["releases"]
    assert_frame_equal(deps1, deps2)
    assert (people1 == people2).all()


@with_defer
async def test_mine_deployments_precomputed_sample(
        sample_deployments, release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps1, people1 = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    deps2, people2 = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, None)
    assert len(deps1) == len(deps2) == 2 * 9
    assert deps1.index.tolist() == deps2.index.tolist()
    lensum = 0
    for i in range(18):
        assert (rel1 := deps1["releases"].iloc[i]).columns.tolist() == \
               (rel2 := deps2["releases"].iloc[i]).columns.tolist(), i
        assert len(rel1) == len(rel2)
        assert (rel1.index.values == rel2.index.values).all()
        lensum += len(rel1)
    assert lensum == 68 * 2
    del deps1["releases"]
    del deps2["releases"]
    assert_frame_equal(deps1, deps2)
    assert (people1 == people2).all()


@with_defer
async def test_mine_deployments_empty(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache):
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps, people = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(deps) == 0


@pytest.mark.parametrize("with_premining", [True, False])
@with_defer
async def test_mine_deployments_event_releases(
        sample_deployments, release_match_setting_event, branches, default_branches,
        prefixer, mdb, pdb, rdb, cache, with_premining):
    await rdb.execute(insert(ReleaseNotification).values(ReleaseNotification(
        account_id=1,
        repository_node_id=40550,
        commit_hash_prefix="1edb992",
        name="Pushed!",
        author_node_id=40020,
        url="www",
        published_at=datetime(2019, 9, 1, tzinfo=timezone.utc),
    ).create_defaults().explode(with_primary_keys=True)))
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    if with_premining:
        await mine_releases(
            ["src-d/go-git"], {}, branches, default_branches, time_from, time_to,
            LabelFilter.empty(), JIRAFilter.empty(), release_match_setting_event,
            LogicalRepositorySettings.empty(),
            prefixer, 1, (6366825,), mdb, pdb, rdb, None, with_avatars=False,
            with_pr_titles=False, with_deployments=False,
        )
        await wait_deferred()
    deps, people = await mine_deployments(
        ["src-d/go-git"], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_event,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    for depname in ("production_2019_11_01", "staging_2019_11_01"):
        df = deps.loc[depname]["releases"]
        assert len(df) == 1
        assert len(df.iloc[0][DeploymentFacts.f.commit_authors]) == 113
        assert len(set(df.iloc[0][DeploymentFacts.f.commit_authors])) == 113
        assert len(df.iloc[0]["prs_node_id"]) == 380
        assert len(set(df.iloc[0]["prs_node_id"])) == 380
        assert df.iloc[0][Release.name.name] == "Pushed!"
        assert df.iloc[0][Release.sha.name] == "1edb992dbc419a0767b1cf3a524b0d35529799f5"


proper_deployments = {
    "Dummy deployment": Deployment(
        name="Dummy deployment",
        conclusion=DeploymentConclusion.SUCCESS,
        environment="production",
        url=None,
        started_at=pd.Timestamp(datetime(2019, 11, 1, 12, 0, tzinfo=timezone.utc)),
        finished_at=pd.Timestamp(datetime(2019, 11, 1, 12, 15, tzinfo=timezone.utc)),
        components=[
            DeployedComponentStruct(
                repository_full_name="src-d/go-git",
                reference="v4.13.1",
                sha="0d1a009cbb604db18be960db5f1525b99a55d727"),
        ],
        labels=None),
}


@with_defer
async def test_mine_release_by_name_deployments(
        release_match_setting_tag_or_branch, prefixer, precomputed_deployments,
        mdb, pdb, rdb):
    names = {"36c78b9d1b1eea682703fb1cbb0f4f3144354389", "v4.0.0"}
    releases, _, deps = await mine_releases_by_name(
        {"src-d/go-git": names},
        release_match_setting_tag_or_branch, LogicalRepositorySettings.empty(), prefixer,
        1, (6366825,), mdb, pdb, rdb, None)
    assert deps == proper_deployments
    assert releases[0][1].deployments == ["Dummy deployment"]


@with_defer
async def test_mine_releases_deployments(
        release_match_setting_tag_or_branch, prefixer, precomputed_deployments,
        branches, default_branches, mdb, pdb, rdb):
    releases, _, _, deps = await mine_releases(
        ["src-d/go-git"], {}, branches, default_branches,
        datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
        LabelFilter.empty(), JIRAFilter.empty(), release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None,
        with_avatars=False, with_pr_titles=False, with_deployments=True)
    assert deps == proper_deployments
    assert len(releases) == 53
    ndeps = 0
    for _, f in releases:
        ndeps += f.deployments is not None and f.deployments[0] == "Dummy deployment"
    assert ndeps == 51
