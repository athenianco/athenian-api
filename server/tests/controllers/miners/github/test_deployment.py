from datetime import datetime, timezone
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from athenian.api.defer import with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.dag_accelerated import (
    extract_independent_ownership,
    extract_pr_commits,
)
from athenian.api.internal.miners.github.release_mine import mine_releases, mine_releases_by_name
from athenian.api.internal.miners.types import (
    DeployedComponent as DeployedComponentStruct,
    Deployment,
    DeploymentConclusion,
)
from athenian.api.internal.settings import LogicalRepositorySettings


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
                sha="0d1a009cbb604db18be960db5f1525b99a55d727",
            ),
        ],
        labels=None,
    ),
}


@with_defer
async def test_mine_release_by_name_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    precomputed_deployments,
    mdb,
    pdb,
    rdb,
):
    names = {"36c78b9d1b1eea682703fb1cbb0f4f3144354389", "v4.0.0"}
    releases, _, deps = await mine_releases_by_name(
        {"src-d/go-git": names},
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert deps == proper_deployments
    assert releases[0][1].deployments == ["Dummy deployment"]


@with_defer
async def test_mine_releases_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    precomputed_deployments,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
):
    releases, _, _, deps = await mine_releases(
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
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
        with_pr_titles=False,
        with_deployments=True,
    )
    assert deps == proper_deployments
    assert len(releases) == 53
    ndeps = 0
    for _, f in releases:
        ndeps += f.deployments is not None and f.deployments[0] == "Dummy deployment"
    assert ndeps == 51
