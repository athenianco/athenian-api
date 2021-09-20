from datetime import datetime, timezone
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import delete, insert

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.dag_accelerated import extract_independent_ownership, \
    extract_pr_commits
from athenian.api.controllers.miners.github.deployment import mine_deployments
from athenian.api.controllers.miners.github.release_mine import mine_releases
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import Release
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel, \
    DeploymentNotification, ReleaseNotification


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
        (2018, 1, 12, "SUCCESS", "4.0.0", 2757510),
        (2018, 1, 11, "FAILURE", "4.0.0", 2757510),
        (2018, 1, 10, "FAILURE", "4.0.0", 2757510),
        (2016, 7, 6, "SUCCESS", "3.1.0", 2756224),
    ):
        for env in ("production", "staging", "canary"):
            name = "%s_%d_%02d_%02d" % (env, year, month, day)
            await rdb.execute(insert(DeploymentNotification).values(dict(
                account_id=1,
                name=name,
                conclusion=conclusion,
                environment=env,
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


async def test_extract_independent_ownership(dag):
    dag = dag["src-d/go-git"]
    ownership = extract_independent_ownership(*dag, np.array([
        b"b65d94e70ea1d013f43234522fa092168e4f1041",
        b"3713157d189a109bdccdb055200defb17297b6de",
        b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff",
        b"0000000000000000000000000000000000000000",
    ], dtype="S40"), np.array([
        b"431af32445562b389397f3ee7af90bf61455fff1",
        b"e80cdbabb92a1ec35ffad536f52d3ff04b548fd1",
        b"0000000000000000000000000000000000000000",
        b"c088fd6a7e1a38e9d5a9815265cb575bb08d08ff",
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


@pytest.mark.xfail
@with_defer
async def test_mine_deployments_from_scratch(
        sample_deployments, release_match_setting_tag_or_branch, branches, default_branches,
        prefixer_promise, mdb, pdb, rdb, cache):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git"], {}, branches, default_branches, time_from, time_to,
        LabelFilter.empty(), JIRAFilter.empty(), release_match_setting_tag_or_branch,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None, with_avatars=False,
    )
    await wait_deferred()
    deps, people = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(deps) == 9 * 2
    pdeps = deps[deps["environment"] == "production"].copy()
    del pdeps["environment"]
    pdeps.sort_index(inplace=True)
    pdeps.reset_index(drop=True, inplace=True)
    sdeps = deps[deps["environment"] == "staging"].copy()
    del sdeps["environment"]
    sdeps.sort_index(inplace=True)
    sdeps.reset_index(drop=True, inplace=True)
    for pdf, sdf in zip(pdeps["releases"].values, sdeps["releases"].values):
        assert_frame_equal(pdf, sdf)
    for df in pdeps["releases"].values:
        assert len(df) == 1
    del pdeps["releases"]
    del sdeps["releases"]
    assert_frame_equal(pdeps, sdeps)


@pytest.mark.skip
@with_defer
async def test_mine_deployments_no_release_facts(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer_promise, mdb, pdb, rdb, cache):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps, people = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(deps) == 1
    assert deps.iloc[0].name == "Dummy deployment"
    obj = deps["releases"].iloc[0].to_dict()
    commit_authors = obj["commit_authors"]
    del obj["commit_authors"]
    assert obj == {
        "additions": {41475: 2},
        "age": {41475: pd.Timedelta("1 days 01:44:14")},
        "author": {41475: "mcuadros"},
        "author_node_id": {41475: 39789},
        "commit_id": {41475: 2755244},
        "commits_count": {41475: 2},
        "deletions": {41475: 2},
        "matched_by": {41475: 1},
        "name": {41475: "v4.13.1"},
        "prs_additions": {41475: np.array([1])},
        "prs_deletions": {41475: np.array([1])},
        "prs_jira": {41475: None},
        "prs_node_id": {41475: np.array([163398])},
        "prs_number": {41475: np.array([1203])},
        "prs_title": {41475: ["worktree: force convert to int64 to support 32bit os. "
                              "Fix #1202"]},
        "prs_user_node_id": {41475: np.array([40187])},
        "published_at": {41475: pd.Timestamp("2019-08-01 15:25:42+0000", tz="UTC")},
        "repository_full_name": {41475: "src-d/go-git"},
        "repository_node_id": {41475: 40550},
        "sha": {41475: "0d1a009cbb604db18be960db5f1525b99a55d727"},
        "tag": {41475: "v4.13.1"},
        "url": {41475: "https://github.com/src-d/go-git/releases/tag/v4.13.1"},
    }
    assert (commit_authors[41475] == np.array([39789, 40187])).all()


@with_defer
async def test_mine_deployments_precomputed_dummy(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer_promise, mdb, pdb, rdb):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps1, people1 = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    deps2, people2 = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        branches, default_branches, prefixer_promise,
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


@pytest.mark.xfail
@with_defer
async def test_mine_deployments_precomputed_sample(
        sample_deployments, release_match_setting_tag_or_branch, branches, default_branches,
        prefixer_promise, mdb, pdb, rdb):
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps1, people1 = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    deps2, people2 = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    assert len(deps1) == len(deps2) == 2 * 9
    assert deps1.index.tolist() == deps2.index.tolist()
    for i in range(18):
        assert (rel1 := deps1["releases"].iloc[i]).columns.tolist() == \
               (rel2 := deps2["releases"].iloc[i]).columns.tolist(), i
        assert len(rel1) == len(rel2) == 1
        assert rel1.index == rel2.index
    del deps1["releases"]
    del deps2["releases"]
    assert_frame_equal(deps1, deps2)
    assert (people1 == people2).all()


@with_defer
async def test_mine_deployments_empty(
        release_match_setting_tag_or_branch, branches, default_branches,
        prefixer_promise, mdb, pdb, rdb, cache):
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    time_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 1, 1, tzinfo=timezone.utc)
    deps, people = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(deps) == 0


@pytest.mark.parametrize("with_premining", [True, False])
@with_defer
async def test_mine_deployments_event_releases(
        sample_deployments, release_match_setting_event, branches, default_branches,
        prefixer_promise, mdb, pdb, rdb, cache, with_premining):
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
            prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None, with_avatars=False,
        )
        await wait_deferred()
    deps, people = await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_event,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, cache)
    for depname in ("production_2019_11_01", "staging_2019_11_01"):
        df = deps.loc[depname]["releases"]
        assert len(df) == 1
        # TODO(vmarkovtsev): if with_premining=False we miss release people
        assert df.iloc[0][Release.name.name] == "Pushed!"
        assert df.iloc[0][Release.sha.name] == "1edb992dbc419a0767b1cf3a524b0d35529799f5"
