from datetime import datetime, timezone

import numpy as np

from athenian.api.defer import with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.release_mine import mine_releases, mine_releases_by_name
from athenian.api.internal.miners.types import (
    DeployedComponent as DeployedComponentStruct,
    Deployment,
    DeploymentConclusion,
    ReleaseFacts,
)
from athenian.api.internal.settings import LogicalRepositorySettings


@with_defer
async def test_mine_release_by_name_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    precomputed_deployments,  # noqa: F811
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
    assert releases.iloc[0][ReleaseFacts.f.deployments] == ["Dummy deployment"]


proper_deployments = {
    "Dummy deployment": Deployment(
        name="Dummy deployment",
        conclusion=DeploymentConclusion.SUCCESS,
        environment="production",
        url=None,
        started_at=np.datetime64(datetime(2019, 11, 1, 12, 0), "us"),
        finished_at=np.datetime64(datetime(2019, 11, 1, 12, 15), "us"),
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
async def test_mine_releases_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    precomputed_deployments,  # noqa: F811
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
        with_extended_pr_details=False,
        with_deployments=True,
    )
    assert deps == proper_deployments
    assert len(releases) == 53
    ndeps = 0
    for rd in releases[ReleaseFacts.f.deployments]:
        ndeps += rd is not None and rd[0] == "Dummy deployment"
    assert ndeps == 51
