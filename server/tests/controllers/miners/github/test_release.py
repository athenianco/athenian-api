from collections import defaultdict
from datetime import datetime, timedelta, timezone
import pickle
from typing import Optional

from freezegun import freeze_time
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import delete, insert, select

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.commit import (
    _empty_dag,
    _fetch_commit_history_dag,
    fetch_repository_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import (
    extract_subdag,
    join_dags,
    mark_dag_access,
    mark_dag_parents,
    partition_dag,
)
from athenian.api.internal.miners.github.release_load import group_repos_by_release_match
from athenian.api.internal.miners.github.release_mine import (
    mine_releases,
    mine_releases_by_name,
    override_first_releases,
)
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.types import PullRequestFacts, ReleaseFacts, released_prs_columns
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
)
from athenian.api.models.metadata.github import Branch, NodeCommit, PullRequest, Release
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, GitHubReleaseFacts
from tests.testutils.db import models_insert
from tests.testutils.factory.persistentdata import ReleaseNotificationFactory
from tests.testutils.time import dt


def check_branch_releases(releases: pd.DataFrame, n: int, time_from: datetime, time_to: datetime):
    assert len(releases) == n
    assert "mcuadros" in set(releases[Release.author.name])
    assert len(releases[Release.commit_id.name].unique()) == n
    assert releases[Release.node_id.name].all()
    assert all(len(n) == 40 for n in releases[Release.name.name])
    assert releases[Release.published_at.name].between(time_from, time_to).all()
    assert (releases[Release.repository_full_name.name] == "src-d/go-git").all()
    assert all(len(n) == 40 for n in releases[Release.sha.name])
    assert len(releases[Release.sha.name].unique()) == n
    assert (~releases[Release.tag.name].values.astype(bool)).all()
    assert releases[Release.url.name].str.startswith("http").all()


@pytest.mark.parametrize("branches_", ["{{default}}", "master", "m.*"])
@with_defer
async def test_load_releases_branches(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    branches_,
    release_loader,
    prefixer,
):
    time_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(branches=branches_)}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}
    check_branch_releases(releases, 240, time_from, time_to)


@with_defer
async def test_load_releases_branches_empty(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_loader,
    prefixer,
):
    time_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(branches="unknown")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert len(releases) == 0
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@pytest.mark.parametrize(
    "time_from, n, pretag",
    [
        (datetime(year=2017, month=10, day=4, tzinfo=timezone.utc), 45, False),
        (datetime(year=2017, month=9, day=4, tzinfo=timezone.utc), 1, False),
        (datetime(year=2017, month=12, day=8, tzinfo=timezone.utc), 0, False),
        (datetime(year=2017, month=9, day=4, tzinfo=timezone.utc), 1, True),
    ],
)
@with_defer
async def test_load_releases_tag_or_branch_dates(
    branches,
    default_branches,
    release_match_setting_tag,
    mdb,
    pdb,
    rdb,
    cache,
    time_from,
    n,
    pretag,
    release_loader,
    prefixer,
):
    time_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)

    if pretag:
        await release_loader.load_releases(
            ["src-d/go-git"],
            branches,
            default_branches,
            time_from,
            time_to,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            rdb,
            cache,
        )
        await wait_deferred()

    release_settings = ReleaseSettings(
        {"github.com/src-d/go-git": _mk_rel_match_settings(branches="master", tags=".*")},
    )
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_settings,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    match_groups, repos_count = group_repos_by_release_match(
        ["src-d/go-git"], default_branches, release_settings,
    )
    spans = (await release_loader.fetch_precomputed_release_match_spans(match_groups, 1, pdb))[
        "src-d/go-git"
    ]
    assert ReleaseMatch.tag in spans
    if n > 1:
        assert ReleaseMatch.branch in spans
        check_branch_releases(releases, n, time_from, time_to)
        assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}
    else:
        assert len(releases) == n
        if n > 0:
            if pretag:
                assert ReleaseMatch.branch not in spans
            assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
        else:
            assert ReleaseMatch.branch in spans
            assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_load_releases_tag_or_branch_initial(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_loader,
    prefixer,
):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2015, month=10, day=22, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(branches="master")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}
    check_branch_releases(releases, 17, time_from, time_to)


@with_defer
async def test_load_releases_tag_logical(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_loader,
    prefixer,
    logical_settings,
    release_match_setting_tag_logical,
):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=10, day=22, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git/alpha", "src-d/go-git/beta"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_tag_logical,
        logical_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert matched_bys == {
        "src-d/go-git/alpha": ReleaseMatch.tag,
        "src-d/go-git/beta": ReleaseMatch.tag,
    }
    assert (releases[Release.repository_full_name.name] == "src-d/go-git/alpha").sum() == 53
    assert (releases[Release.repository_full_name.name] == "src-d/go-git/beta").sum() == 37
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git", "src-d/go-git/alpha", "src-d/go-git/beta"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_tag_logical,
        logical_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert matched_bys == {
        "src-d/go-git": ReleaseMatch.tag,
        "src-d/go-git/alpha": ReleaseMatch.tag,
        "src-d/go-git/beta": ReleaseMatch.tag,
    }
    assert (releases[Release.repository_full_name.name] == "src-d/go-git").sum() == 53
    assert (releases[Release.repository_full_name.name] == "src-d/go-git/alpha").sum() == 53
    assert (releases[Release.repository_full_name.name] == "src-d/go-git/beta").sum() == 37
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git", "src-d/go-git/beta"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_tag_logical,
        logical_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert matched_bys == {
        "src-d/go-git": ReleaseMatch.tag,
        "src-d/go-git/beta": ReleaseMatch.tag,
    }
    assert (releases[Release.repository_full_name.name] == "src-d/go-git").sum() == 53
    assert (releases[Release.repository_full_name.name] == "src-d/go-git/beta").sum() == 37


@with_defer
async def test_load_releases_events_logical(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_loader,
    prefixer,
    logical_settings,
):
    await models_insert(
        rdb,
        ReleaseNotificationFactory(
            repository_node_id=40550, published_at=dt(2019, 2, 2), commit_hash_prefix="8d20cc5",
        ),
    )
    release_settings = ReleaseSettings(
        {
            "github.com/src-d/go-git/alpha": _mk_rel_match_settings(events=".*"),
            "github.com/src-d/go-git/beta": _mk_rel_match_settings(tags=".*~beta.*"),
        },
    )
    releases, _ = await release_loader.load_releases(
        ["src-d/go-git/alpha", "src-d/go-git/beta"],
        branches,
        default_branches,
        dt(2015, 1, 1),
        dt(2020, 10, 22),
        release_settings,
        logical_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    # only release for logical repo alpha has been created, beta has "tag" release settings
    assert list(releases[Release.repository_full_name.name]) == ["src-d/go-git/alpha"]
    assert list(releases[Release.repository_node_id.name]) == [40550]


@with_defer
async def test_load_releases_events_logical_release_name_match(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_loader,
    prefixer,
    logical_settings,
):
    await models_insert(
        rdb,
        ReleaseNotificationFactory(
            repository_node_id=40550,
            published_at=dt(2019, 2, 2),
            commit_hash_prefix="8d20cc5",
            name="alpha-1.1.1",
        ),
    )
    release_settings = ReleaseSettings(
        {
            "github.com/src-d/go-git/alpha": _mk_rel_match_settings(events="^alpha.*"),
            "github.com/src-d/go-git/beta": _mk_rel_match_settings(events="^beta.*"),
        },
    )
    releases, _ = await release_loader.load_releases(
        ["src-d/go-git/alpha", "src-d/go-git/beta"],
        branches,
        default_branches,
        dt(2015, 1, 1),
        dt(2020, 10, 22),
        release_settings,
        logical_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    # only events release setting for repo alpha matches
    assert list(releases[Release.repository_full_name.name]) == ["src-d/go-git/alpha"]
    assert list(releases[Release.repository_node_id.name]) == [40550]


@with_defer
async def test_map_releases_to_prs_branches(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    releases_to_prs_mapper,
    prefixer,
):
    time_from = datetime(year=2015, month=4, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2015, month=5, day=1, tzinfo=timezone.utc)
    release_settings = ReleaseSettings(
        {"github.com/src-d/go-git": _mk_rel_match_settings(branches="master")},
    )
    (
        prs,
        releases,
        new_settings,
        matched_bys,
        _,
        _,
    ) = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        [],
        [],
        JIRAFilter.empty(),
        release_settings,
        LogicalRepositorySettings.empty(),
        None,
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert prs.empty
    assert len(releases) == 1
    assert releases[Release.sha.name][0] == b"5d7303c49ac984a9fec60523f2d5297682e16646"
    assert new_settings == release_settings
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_map_releases_to_prs_updated_min_max(
    branches,
    default_branches,
    release_match_setting_tag,
    mdb,
    pdb,
    rdb,
    releases_to_prs_mapper,
    prefixer,
):
    prs, releases, _, matched_bys, _, _ = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2018, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2030, month=12, day=2, tzinfo=timezone.utc),
        [],
        [],
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        datetime(2018, 7, 20, tzinfo=timezone.utc),
        datetime(2019, 1, 1, tzinfo=timezone.utc),
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        truncate=False,
    )
    assert len(prs) == 5
    assert releases is not None
    assert len(releases) == 12


@pytest.mark.parametrize("repos", [["src-d/gitbase"], []])
@with_defer
async def test_load_releases_empty(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    repos,
    release_loader,
    prefixer,
):
    releases, matched_bys = await release_loader.load_releases(
        repos,
        branches,
        default_branches,
        datetime(year=2020, month=6, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/gitbase": _mk_rel_match_settings(branches=".*")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.node_id.name,
    )
    assert releases.empty
    if repos:
        assert matched_bys == {"src-d/gitbase": ReleaseMatch.branch}
    time_from = datetime(year=2017, month=3, day=4, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(tags="")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert releases.empty
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(branches="")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert releases.empty
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_load_releases_events_settings(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_loader,
    prefixer,
):
    await rdb.execute(
        insert(ReleaseNotification).values(
            ReleaseNotification(
                account_id=1,
                repository_node_id=40550,
                commit_hash_prefix="8d20cc5",
                name="Pushed!",
                author_node_id=40020,
                url="www",
                published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    releases, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2019, month=1, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(tags=".*")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.node_id.name,
    )
    await wait_deferred()
    assert len(releases) == 7
    assert (releases[matched_by_column] == ReleaseMatch.tag).all()
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2019, month=1, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(events=".*")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.node_id.name,
    )
    await wait_deferred()
    assert matched_bys == {"src-d/go-git": ReleaseMatch.event}
    assert (releases[matched_by_column] == ReleaseMatch.event).all()
    assert len(releases) == 1
    assert releases.index[0] == 2756775
    assert releases.iloc[0].to_dict() == {
        Release.repository_full_name.name: "src-d/go-git",
        Release.repository_node_id.name: 40550,
        Release.author.name: "vmarkovtsev",
        Release.author_node_id.name: 40020,
        Release.name.name: "Pushed!",
        Release.published_at.name: pd.Timestamp("2020-01-01 00:00:00", tzinfo=timezone.utc),
        Release.tag.name: None,
        Release.url.name: "www",
        Release.sha.name: b"8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        Release.commit_id.name: 2756775,
        matched_by_column: ReleaseMatch.event,
    }
    rows = await rdb.fetch_all(select([ReleaseNotification]))
    assert len(rows) == 1
    values = dict(rows[0])
    assert (
        values[ReleaseNotification.updated_at.name] > values[ReleaseNotification.created_at.name]
    )
    del values[ReleaseNotification.updated_at.name]
    del values[ReleaseNotification.created_at.name]
    if rdb.url.dialect == "sqlite":
        tzinfo = None
    else:
        tzinfo = timezone.utc
    assert values == {
        "account_id": 1,
        "repository_node_id": 40550,
        "commit_hash_prefix": "8d20cc5",
        "resolved_commit_hash": "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        "resolved_commit_node_id": 2756775,  # noqa
        "name": "Pushed!",
        "author_node_id": 40020,
        "url": "www",
        "published_at": datetime(2020, 1, 1, tzinfo=tzinfo),
        "cloned": False,
    }


@with_defer
async def test_load_releases_events_unresolved(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_loader,
    prefixer,
):
    await rdb.execute(
        insert(ReleaseNotification).values(
            ReleaseNotification(
                account_id=1,
                repository_node_id=40550,
                commit_hash_prefix="whatever",
                name="Pushed!",
                author_node_id=40020,
                url="www",
                published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2019, month=1, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/go-git": _mk_rel_match_settings(events=".*")}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.node_id.name,
    )
    assert releases.empty
    assert matched_bys == {"src-d/go-git": ReleaseMatch.event}


@pytest.fixture(scope="module")
def heads_df1():
    df = pd.DataFrame(
        [(b"5d7303c49ac984a9fec60523f2d5297682e16646", 2756216, 525, "src-d/go-git")],
        columns=["1", "2", "3", "4"],
    )
    df["1"] = df["1"].values.astype("S40")
    return df


@pytest.fixture(scope="module")
def heads_df2():
    df = pd.DataFrame(
        [
            (b"d2a38b4a5965d529566566640519d03d2bd10f6c", 2757677, 525, "src-d/go-git"),
            (b"31eae7b619d166c366bf5df4991f04ba8cebea0a", 2755667, 611, "src-d/go-git"),
        ],
        columns=["1", "2", "3", "4"],
    )
    df["1"] = df["1"].values.astype("S40")
    return df


@pytest.mark.parametrize("prune", [False, True])
@with_defer
async def test__fetch_repository_commits_smoke(mdb, pdb, prune, heads_df2):
    dags = await fetch_repository_commits(
        {"src-d/go-git": (True, _empty_dag())},
        heads_df2,
        ("1", "2", "3", "4"),
        prune,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    assert isinstance(dags, dict)
    assert len(dags) == 1
    hashes, vertexes, edges = dags["src-d/go-git"][1]
    ground_truth = {
        "31eae7b619d166c366bf5df4991f04ba8cebea0a": [
            "b977a025ca21e3b5ca123d8093bd7917694f6da7",
            "d2a38b4a5965d529566566640519d03d2bd10f6c",
        ],
        "b977a025ca21e3b5ca123d8093bd7917694f6da7": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "d2a38b4a5965d529566566640519d03d2bd10f6c": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "35b585759cbf29f8ec428ef89da20705d59f99ec": ["c2bbf9fe8009b22d0f390f3c8c3f13937067590f"],
        "c2bbf9fe8009b22d0f390f3c8c3f13937067590f": ["fc9f0643b21cfe571046e27e0c4565f3a1ee96c8"],
        "fc9f0643b21cfe571046e27e0c4565f3a1ee96c8": ["c088fd6a7e1a38e9d5a9815265cb575bb08d08ff"],
        "c088fd6a7e1a38e9d5a9815265cb575bb08d08ff": ["5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf"],
        "5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf": ["5d7303c49ac984a9fec60523f2d5297682e16646"],
        "5d7303c49ac984a9fec60523f2d5297682e16646": [],
    }
    for k, v in ground_truth.items():
        vertex = np.where(hashes == k.encode())[0][0]
        assert hashes[edges[vertexes[vertex] : vertexes[vertex + 1]]].astype("U40").tolist() == v
    assert len(hashes) == 9
    await wait_deferred()
    dags2 = await fetch_repository_commits(
        dags,
        heads_df2,
        ("1", "2", "3", "4"),
        prune,
        1,
        (6366825,),
        Database("sqlite://"),
        pdb,
        None,
    )
    assert pickle.dumps(dags2) == pickle.dumps(dags)
    heads_df = heads_df2.copy(deep=True)
    heads_df["1"].values[0] = b"1353ccd6944ab41082099b79979ded3223db98ec"
    heads_df["2"].values[0] = 2755667
    with pytest.raises(Exception):
        await fetch_repository_commits(
            dags,
            heads_df,
            ("1", "2", "3", "4"),
            prune,
            1,
            (6366825,),
            Database("sqlite://"),
            pdb,
            None,
        )


@pytest.mark.parametrize("prune", [False, True])
@with_defer
async def test__fetch_repository_commits_initial_commit(mdb, pdb, prune, heads_df1):
    dags = await fetch_repository_commits(
        {"src-d/go-git": (True, _empty_dag())},
        heads_df1,
        ("1", "2", "3", "4"),
        prune,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    consistent, (hashes, vertexes, edges) = dags["src-d/go-git"]
    assert consistent
    assert hashes == np.array(["5d7303c49ac984a9fec60523f2d5297682e16646"], dtype="S40")
    assert (vertexes == np.array([0, 0], dtype=np.uint32)).all()
    assert (edges == np.array([], dtype=np.uint32)).all()


@pytest.mark.parametrize("prune", [False, True])
@with_defer
@freeze_time("2015-04-05")
async def test__fetch_repository_commits_orphan_skip(mdb, pdb, prune, heads_df1):
    dags = await fetch_repository_commits(
        {
            "src-d/go-git": (
                True,
                (
                    np.array(["7" * 40], dtype="S40"),
                    np.array([0, 0], dtype=np.uint32),
                    np.array([], dtype=np.uint32),
                ),
            ),
        },
        heads_df1,
        ("1", "2", "3", "4"),
        prune,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    hashes, vertexes, edges = dags["src-d/go-git"][1]
    if prune:
        assert len(hashes) == 0
    else:
        assert hashes == np.array(["7" * 40], dtype="S40")


@pytest.mark.parametrize("prune", [False, True])
@with_defer
async def test__fetch_repository_commits_orphan_include(mdb, pdb, prune, heads_df1):
    dags = await fetch_repository_commits(
        {
            "src-d/go-git": (
                True,
                (
                    np.array(["7" * 40], dtype="S40"),
                    np.array([0, 0], dtype=np.uint32),
                    np.array([], dtype=np.uint32),
                ),
            ),
        },
        heads_df1,
        ("1", "2", "3", "4"),
        prune,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    hashes, vertexes, edges = dags["src-d/go-git"][1]
    if prune:
        assert hashes == np.array(["5d7303c49ac984a9fec60523f2d5297682e16646"], dtype="S40")
    else:
        assert_array_equal(
            hashes, np.array(["5d7303c49ac984a9fec60523f2d5297682e16646", "7" * 40], dtype="S40"),
        )


@with_defer
async def test__fetch_repository_commits_cache(mdb, pdb, cache, heads_df2):
    dags1 = await fetch_repository_commits(
        {"src-d/go-git": (True, _empty_dag())},
        heads_df2,
        ("1", "2", "3", "4"),
        False,
        1,
        (6366825,),
        mdb,
        pdb,
        cache,
    )
    await wait_deferred()
    dags2 = await fetch_repository_commits(
        {"src-d/go-git": (True, _empty_dag())},
        heads_df2,
        ("1", "2", "3", "4"),
        False,
        1,
        (6366825,),
        None,
        None,
        cache,
    )
    assert pickle.dumps(dags1) == pickle.dumps(dags2)
    fake_pdb = Database("sqlite://")

    class FakeMetrics:
        def get(self):
            return defaultdict(int)

    fake_pdb.metrics = {"hits": FakeMetrics(), "misses": FakeMetrics()}
    with pytest.raises(Exception):
        await fetch_repository_commits(
            {"src-d/go-git": (True, _empty_dag())},
            heads_df2,
            ("1", "2", "3", "4"),
            True,
            1,
            (6366825,),
            None,
            fake_pdb,
            cache,
        )


@with_defer
async def test__fetch_repository_commits_many(mdb, pdb, heads_df2):
    dags = await fetch_repository_commits(
        {"src-d/go-git": (True, _empty_dag())},
        heads_df2,
        ("1", "2", "3", "4"),
        False,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    assert len(dags["src-d/go-git"][1][0]) == 9


@with_defer
async def test__fetch_repository_commits_full(mdb, pdb, dag, cache, branch_miner, prefixer):
    branches, _ = await branch_miner.extract_branches(dag, prefixer, (6366825,), mdb, None)
    commit_ids = branches[Branch.commit_id.name].values
    commit_dates = await mdb.fetch_all(
        select([NodeCommit.id, NodeCommit.committed_date]).where(NodeCommit.id.in_(commit_ids)),
    )
    commit_dates = {r[0]: r[1] for r in commit_dates}
    if mdb.url.dialect == "sqlite":
        commit_dates = {k: v.replace(tzinfo=timezone.utc) for k, v in commit_dates.items()}
    now = datetime.now(timezone.utc)
    branches[Branch.commit_date] = [commit_dates.get(commit_id, now) for commit_id in commit_ids]
    cols = (
        Branch.commit_sha.name,
        Branch.commit_id.name,
        Branch.commit_date,
        Branch.repository_full_name.name,
    )
    commits = await fetch_repository_commits(
        dag, branches, cols, False, 1, (6366825,), mdb, pdb, cache,
    )
    await wait_deferred()
    assert len(commits) == 1
    assert len(commits["src-d/go-git"][1][0]) == 1919
    branches = branches[branches[Branch.branch_name.name] == "master"]
    commits = await fetch_repository_commits(
        commits, branches, cols, False, 1, (6366825,), mdb, pdb, cache,
    )
    await wait_deferred()
    assert len(commits) == 1
    assert len(commits["src-d/go-git"][1][0]) == 1919  # with force-pushed commits
    commits = await fetch_repository_commits(
        commits, branches, cols, True, 1, (6366825,), mdb, pdb, cache,
    )
    await wait_deferred()
    assert len(commits) == 1
    assert len(commits["src-d/go-git"][1][0]) == 1538  # without force-pushed commits


@with_defer
async def test__fetch_repository_first_commit_dates_pdb_cache(
    mdb,
    pdb,
    cache,
    releases_to_prs_mapper,
):
    fcd1 = await releases_to_prs_mapper._fetch_repository_first_commit_dates(
        ["src-d/go-git"], 1, (6366825,), mdb, pdb, cache,
    )
    await wait_deferred()
    fcd2 = await releases_to_prs_mapper._fetch_repository_first_commit_dates(
        ["src-d/go-git"], 1, (6366825,), Database("sqlite://"), pdb, None,
    )
    fcd3 = await releases_to_prs_mapper._fetch_repository_first_commit_dates(
        ["src-d/go-git"], 1, (6366825,), Database("sqlite://"), Database("sqlite://"), cache,
    )
    assert len(fcd1) == len(fcd2) == len(fcd3) == 1
    assert fcd1["src-d/go-git"] == fcd2["src-d/go-git"] == fcd3["src-d/go-git"]
    assert fcd1["src-d/go-git"].tzinfo == timezone.utc


def test_extract_subdag_smoke():
    hashes = np.array(
        [
            "308a9f90707fb9d12cbcd28da1bc33da436386fe",
            "33cafc14532228edca160e46af10341a8a632e3e",
            "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
            "a444ccadf5fddad6ad432c13a239c74636c7f94f",
        ],
        dtype="S40",
    )
    vertexes = np.array([0, 1, 2, 3, 3], dtype=np.uint32)
    edges = np.array([3, 0, 0], dtype=np.uint32)
    heads = np.array(["61a719e0ff7522cc0d129acb3b922c94a8a5dbca"], dtype="S40")
    new_hashes, new_vertexes, new_edges = extract_subdag(hashes, vertexes, edges, heads)
    assert (
        new_hashes
        == np.array(
            [
                "308a9f90707fb9d12cbcd28da1bc33da436386fe",
                "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                "a444ccadf5fddad6ad432c13a239c74636c7f94f",
            ],
            dtype="S40",
        )
    ).all()
    assert (new_vertexes == np.array([0, 1, 2, 2], dtype=np.uint32)).all()
    assert (new_edges == np.array([2, 0], dtype=np.uint32)).all()


def test_extract_subdag_empty():
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    heads = np.array(["61a719e0ff7522cc0d129acb3b922c94a8a5dbca"], dtype="S40")
    new_hashes, new_vertexes, new_edges = extract_subdag(hashes, vertexes, edges, heads)
    assert len(new_hashes) == 0
    assert (new_vertexes == vertexes).all()
    assert len(new_edges) == 0


def test_join_dags_smoke():
    hashes = np.array(
        [
            "308a9f90707fb9d12cbcd28da1bc33da436386fe",
            "33cafc14532228edca160e46af10341a8a632e3e",
            "a444ccadf5fddad6ad432c13a239c74636c7f94f",
        ],
        dtype="S40",
    )
    vertexes = np.array([0, 1, 2, 2], dtype=np.uint32)
    edges = np.array([2, 0], dtype=np.uint32)
    new_hashes, new_vertexes, new_edges = join_dags(
        hashes,
        vertexes,
        edges,
        [
            (
                "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                "308a9f90707fb9d12cbcd28da1bc33da436386fe",
                0,
            ),
            (
                "308a9f90707fb9d12cbcd28da1bc33da436386fe",
                "a444ccadf5fddad6ad432c13a239c74636c7f94f",
                0,
            ),
            (
                "8d27ef15cc9b334667d8adc9ce538222c5ac3607",
                "33cafc14532228edca160e46af10341a8a632e3e",
                1,
            ),
            (
                "8d27ef15cc9b334667d8adc9ce538222c5ac3607",
                "308a9f90707fb9d12cbcd28da1bc33da436386fe",
                0,
            ),
        ],
    )
    assert (
        new_hashes
        == np.array(
            [
                "308a9f90707fb9d12cbcd28da1bc33da436386fe",
                "33cafc14532228edca160e46af10341a8a632e3e",
                "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                "8d27ef15cc9b334667d8adc9ce538222c5ac3607",
                "a444ccadf5fddad6ad432c13a239c74636c7f94f",
            ],
            dtype="S40",
        )
    ).all()
    assert (new_vertexes == np.array([0, 1, 2, 3, 5, 5], dtype=np.uint32)).all()
    assert (new_edges == np.array([4, 0, 0, 0, 1], dtype=np.uint32)).all()


def test_mark_dag_access_smoke():
    hashes = np.array(
        [
            "308a9f90707fb9d12cbcd28da1bc33da436386fe",
            "33cafc14532228edca160e46af10341a8a632e3e",
            "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
            "a444ccadf5fddad6ad432c13a239c74636c7f94f",
        ],
        dtype="S40",
    )
    # 33cafc14532228edca160e46af10341a8a632e3e -> 308a9f90707fb9d12cbcd28da1bc33da436386fe
    # 33cafc14532228edca160e46af10341a8a632e3e -> 61a719e0ff7522cc0d129acb3b922c94a8a5dbca
    # 61a719e0ff7522cc0d129acb3b922c94a8a5dbca -> 308a9f90707fb9d12cbcd28da1bc33da436386fe
    # 308a9f90707fb9d12cbcd28da1bc33da436386fe -> a444ccadf5fddad6ad432c13a239c74636c7f94f
    vertexes = np.array([0, 1, 3, 4, 4], dtype=np.uint32)
    edges = np.array([3, 0, 2, 0], dtype=np.uint32)
    heads = np.array(
        ["33cafc14532228edca160e46af10341a8a632e3e", "61a719e0ff7522cc0d129acb3b922c94a8a5dbca"],
        dtype="S40",
    )
    marks = mark_dag_access(hashes, vertexes, edges, heads, True)
    assert_array_equal(marks, np.array([1, 0, 1, 1], dtype=np.int32))
    heads = np.array(
        ["61a719e0ff7522cc0d129acb3b922c94a8a5dbca", "33cafc14532228edca160e46af10341a8a632e3e"],
        dtype="S40",
    )
    # 33cafc14532228edca160e46af10341a8a632e3e shows the oldest, but it's the entry => takes all
    marks = mark_dag_access(hashes, vertexes, edges, heads, True)
    assert_array_equal(marks, np.array([1, 1, 1, 1], dtype=np.int32))
    marks = mark_dag_access(hashes, vertexes, edges, heads, False)
    assert_array_equal(marks, np.array([0, 1, 0, 0], dtype=np.int32))


def test_mark_dag_access_empty():
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    heads = np.array(
        ["33cafc14532228edca160e46af10341a8a632e3e", "61a719e0ff7522cc0d129acb3b922c94a8a5dbca"],
        dtype="S40",
    )
    marks = mark_dag_access(hashes, vertexes, edges, heads, True)
    assert len(marks) == 0


async def test_partition_dag(dag):
    hashes, vertexes, edges = dag["src-d/go-git"][1]
    p = partition_dag(hashes, vertexes, edges, [b"ad9456267524e08efcf4486cadfb6cef8d182677"])
    assert p.tolist() == [b"ad9456267524e08efcf4486cadfb6cef8d182677"]
    p = partition_dag(hashes, vertexes, edges, [b"7cd021554eb318165dd28988fe1675a5e5c32601"])
    assert p.tolist() == [
        b"7cd021554eb318165dd28988fe1675a5e5c32601",
        b"ced875aec7bef9113e1c37b1b811a59e17dbd138",
    ]


def test_partition_dag_empty():
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    p = partition_dag(hashes, vertexes, edges, ["ad9456267524e08efcf4486cadfb6cef8d182677"])
    assert len(p) == 0


async def test__fetch_commit_history_dag_stops(mdb, dag):
    hashes, vertexes, edges = dag["src-d/go-git"][1]
    subhashes, subvertexes, subedges = extract_subdag(
        hashes,
        vertexes,
        edges,
        np.array([b"364866fc77fac656e103c1048dd7da4764c6d9d9"], dtype="S40"),
    )
    assert len(subhashes) == 1414
    _, _, newhashes, newvertexes, newedges = await _fetch_commit_history_dag(
        subhashes,
        subvertexes,
        subedges,
        ["f6305131a06bd94ef39e444b60f773db75b054f6"],
        [2755363],
        "src-d/go-git",
        (6366825,),
        mdb,
    )
    assert_array_equal(newhashes, hashes)
    assert_array_equal(newvertexes, vertexes)
    assert_array_equal(newedges, edges)
    _, _, newhashes, newvertexes, newedges = await _fetch_commit_history_dag(
        subhashes,
        subvertexes,
        subedges,
        ["1a7db85bca7027d90afdb5ce711622aaac9feaed"],
        [2755363],
        "src-d/go-git",
        (6366825,),
        mdb,
    )
    assert_array_equal(newhashes, hashes)
    assert_array_equal(newvertexes, vertexes)
    assert_array_equal(newedges, edges)


@with_defer
async def test_mark_dag_parents_smoke(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    dag,
    release_loader,
    prefixer,
):
    hashes, vertexes, edges = dag["src-d/go-git"][1]
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    release_hashes = releases[Release.sha.name].values
    release_dates = releases[Release.published_at.name].values
    ownership = mark_dag_access(hashes, vertexes, edges, release_hashes, True)
    parents = mark_dag_parents(hashes, vertexes, edges, release_hashes, release_dates, ownership)
    array = np.array
    uint32 = np.uint32
    ground_truth = array(
        [
            array([1], dtype=uint32),
            array([2], dtype=uint32),
            array([3], dtype=uint32),
            array([4], dtype=uint32),
            array([5, 8, 9], dtype=uint32),
            array([6], dtype=uint32),
            array([7], dtype=uint32),
            array([8], dtype=uint32),
            array([9], dtype=uint32),
            array([10, 11, 14, 19], dtype=uint32),
            array([11, 12], dtype=uint32),
            array([12], dtype=uint32),
            array([13], dtype=uint32),
            array([14], dtype=uint32),
            array([15], dtype=uint32),
            array([16], dtype=uint32),
            array([17], dtype=uint32),
            array([18], dtype=uint32),
            array([19], dtype=uint32),
            array([20], dtype=uint32),
            array([21], dtype=uint32),
            array([22, 23], dtype=uint32),
            array([23], dtype=uint32),
            array([24], dtype=uint32),
            array([25], dtype=uint32),
            array([26], dtype=uint32),
            array([27], dtype=uint32),
            array([28], dtype=uint32),
            array([29], dtype=uint32),
            array([30], dtype=uint32),
            array([31], dtype=uint32),
            array([32], dtype=uint32),
            array([34], dtype=uint32),
            array([], dtype=uint32),
            array([35], dtype=uint32),
            array([36], dtype=uint32),
            array([37], dtype=uint32),
            array([38], dtype=uint32),
            array([39], dtype=uint32),
            array([40], dtype=uint32),
            array([41], dtype=uint32),
            array([42], dtype=uint32),
            array([43], dtype=uint32),
            array([44], dtype=uint32),
            array([46, 47], dtype=uint32),
            array([], dtype=uint32),
            array([47], dtype=uint32),
            array([48], dtype=uint32),
            array([49], dtype=uint32),
            array([50], dtype=uint32),
            array([51], dtype=uint32),
            array([52], dtype=uint32),
            array([], dtype=uint32),
        ],
        dtype=object,
    )
    for yours, mine in zip(parents, ground_truth):
        assert (yours == mine).all()


@with_defer
async def test_mark_dag_parents_empty(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    release_loader,
    prefixer,
):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    release_hashes = releases[Release.sha.name].values
    release_dates = releases[Release.published_at.name].values
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    ownership = mark_dag_access(hashes, vertexes, edges, release_hashes, True)
    parents = mark_dag_parents(hashes, vertexes, edges, release_hashes, release_dates, ownership)
    assert len(parents) == len(release_hashes)
    for p in parents:
        assert p == []


class TestMineReleases:
    @with_defer
    async def test_full_span(self, mdb, pdb, rdb, release_match_setting_tag, prefixer):
        time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )
        releases, avatars, matched_bys, _ = await mine_releases(**kwargs)
        assert len(releases) == 53
        assert len(avatars) == 124
        assert matched_bys == {"github.com/src-d/go-git": ReleaseMatch.tag}
        for details, facts in releases:
            assert details[Release.name.name]
            assert details[Release.url.name]
            assert details[Release.repository_full_name.name] == "github.com/src-d/go-git"
            assert len(facts.commit_authors) > 0
            assert all(a >= 0 for a in facts.commit_authors)
            assert facts.age
            assert facts.publisher >= 0
            assert facts.additions > 0
            assert facts.deletions > 0
            assert facts.commits_count > 0
            assert len(facts["prs_" + PullRequest.number.name]) or facts.published <= pd.Timestamp(
                "2017-02-01 09:51:10",
            )
            assert time_from < facts.published.item().replace(tzinfo=timezone.utc) < time_to
            assert facts.matched_by == ReleaseMatch.tag

    @with_defer
    async def test_precomputed_smoke(
        self,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag,
        release_match_setting_branch,
        prefixer,
        branches,
        default_branches,
    ):
        time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
        )
        await mine_releases(**kwargs)
        await wait_deferred()
        kwargs["with_deployments"] = False
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 53
        assert len(avatars) == 124

        kwargs = self._kwargs(
            time_from=time_to,
            time_to=time_to,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 0

        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            branches=branches,
            default_branches=default_branches,
            release_settings=release_match_setting_branch,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 772
        assert len(avatars) == 131
        await wait_deferred()

        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 772
        assert len(avatars) == 131

    @with_defer
    async def test_precomputed_time_range(
        self,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag,
        prefixer,
    ):
        time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)

        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )

        releases, avatars, _, _ = await mine_releases(**kwargs)
        await wait_deferred()

        kwargs["time_from"] = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
        kwargs["time_to"] = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
        releases, avatars, _, _ = await mine_releases(**kwargs)
        for _, f in releases:
            assert time_from <= f.published.item().replace(tzinfo=timezone.utc) < time_to
            for col in released_prs_columns(PullRequest):
                assert len(f["prs_" + col.name]) > 0
            assert f.commits_count > 0
        assert len(releases) == 22
        assert len(avatars) == 93

    @with_defer
    async def test_precomputed_update(self, mdb, pdb, rdb, release_match_setting_tag, prefixer):
        time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2018, month=11, day=1, tzinfo=timezone.utc)
        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        await wait_deferred()

        kwargs["time_from"] = time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
        kwargs["time_to"] = time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
        releases, avatars, _, _ = await mine_releases(**kwargs)
        for _, f in releases:
            assert time_from <= f.published.item().replace(tzinfo=timezone.utc) < time_to
            assert len(getattr(f, "prs_" + PullRequest.number.name)) > 0
            assert f.commits_count > 0
        assert len(releases) == 22
        assert len(avatars) == 93

    @with_defer
    async def test_jira(self, mdb, pdb, rdb, release_match_setting_tag, prefixer, cache):
        time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2020, month=11, day=1, tzinfo=timezone.utc)
        jira_filter = JIRAFilter(
            1,
            ["10003", "10009"],
            LabelFilter({"bug", "onboarding", "performance"}, set()),
            set(),
            set(),
            False,
            False,
        )
        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            jira=jira_filter,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        await wait_deferred()
        assert len(releases) == 8

        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )

        releases, avatars, _, _ = await mine_releases(**kwargs)
        await wait_deferred()
        assert len(releases) == 22

        jira_filter = JIRAFilter(
            1,
            ["10003", "10009"],
            LabelFilter({"bug", "onboarding", "performance"}, set()),
            set(),
            set(),
            False,
            False,
        )
        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            jira=jira_filter,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
            cache=cache,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 8
        await wait_deferred()

        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            cache=cache,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 22
        jira_filter = JIRAFilter(
            1, ["10003", "10009"], LabelFilter.empty(), set(), set(), False, True,
        )
        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            jira=jira_filter,
            release_settings=release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
            cache=cache,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 15

    @with_defer
    async def test_logical_jira(
        self,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag_logical,
        logical_settings,
        prefixer,
        cache,
    ):
        time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2020, month=11, day=1, tzinfo=timezone.utc)
        repo = "src-d/go-git/alpha"
        jira_filter = JIRAFilter(
            1,
            ["10003", "10009"],
            LabelFilter({"bug", "onboarding", "performance"}, set()),
            set(),
            set(),
            False,
            False,
        )
        kwargs = self._kwargs(
            repos=[repo],
            time_from=time_from,
            time_to=time_to,
            logical_settings=logical_settings,
            release_settings=release_match_setting_tag_logical,
            jira=jira_filter,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            with_deployments=False,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        await wait_deferred()
        assert len(releases) == 4
        kwargs["jira"] = JIRAFilter.empty()
        releases, avatars, _, _ = await mine_releases(**kwargs)
        await wait_deferred()
        assert len(releases) == 22

        kwargs["jira"] = jira_filter
        kwargs["cache"] = cache
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 4
        await wait_deferred()

        kwargs["jira"] = JIRAFilter.empty()
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 22
        kwargs["jira"] = JIRAFilter(
            1, ["10003", "10009"], LabelFilter.empty(), set(), set(), False, True,
        )
        releases, avatars, _, _ = await mine_releases(**kwargs)
        assert len(releases) == 12

    @classmethod
    def _kwargs(cls, **extra):
        extra.setdefault("repos", ["src-d/go-git"])
        extra.setdefault("participants", {})
        extra.setdefault("branches", None)
        extra.setdefault("default_branches", {})
        extra.setdefault("labels", LabelFilter.empty())
        extra.setdefault("jira", JIRAFilter.empty())
        extra.setdefault("logical_settings", LogicalRepositorySettings.empty())
        extra.setdefault("account", 1)
        extra.setdefault("meta_ids", (6366825,))
        extra.setdefault("cache", None)

        return extra


@with_defer
async def test_mine_releases_labels(mdb, pdb, rdb, release_match_setting_tag, prefixer, cache):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=11, day=1, tzinfo=timezone.utc)
    releases1, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
        with_avatars=False,
        with_pr_titles=False,
        with_deployments=False,
    )
    await wait_deferred()
    assert len(releases1) == 22
    releases2, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter({"bug", "enhancement", "plumbing"}, set()),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
        with_avatars=False,
        with_pr_titles=False,
        with_deployments=False,
    )
    assert len(releases2) == 3
    releases3, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter(set(), {"bug", "enhancement", "plumbing"}),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
        with_avatars=False,
        with_pr_titles=False,
        with_deployments=False,
    )
    assert len(releases3) == 22
    reduced = defaultdict(int)
    for (det1, facts1), (det2, facts2) in zip(releases1, releases3):
        assert det1 == det2
        for col in released_prs_columns(PullRequest):
            key = "prs_" + col.name
        reduced[key] += len(facts1[key]) > len(facts2[key])
    vals = np.array(list(reduced.values()))
    assert (vals == vals[0]).all()
    assert vals[0] == 3


@with_defer
async def test_mine_releases_cache(mdb, pdb, rdb, release_match_setting_tag, prefixer, cache):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2018, month=11, day=1, tzinfo=timezone.utc)
    releases1, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
        with_avatars=False,
        with_pr_titles=False,
        with_deployments=False,
    )
    await wait_deferred()
    releases2, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
        with_avatars=False,
        with_pr_titles=False,
        with_deployments=False,
    )
    assert releases1 == releases2
    with pytest.raises(AssertionError):
        await mine_releases(
            ["src-d/go-git"],
            {},
            None,
            {},
            time_from,
            time_to,
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            1,
            (6366825,),
            None,
            None,
            None,
            cache,
            with_pr_titles=True,
        )
    releases3, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
        with_pr_titles=True,
        with_avatars=False,
        with_deployments=False,
    )
    await wait_deferred()
    releases4, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
        with_pr_titles=True,
        with_avatars=False,
        with_deployments=False,
    )
    assert releases3 == releases4
    releases2, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        {},
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
        with_pr_titles=False,
        with_avatars=False,
        with_deployments=False,
    )
    assert releases3 == releases2


@with_defer
async def test_mine_releases_logical_title(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag_logical,
    prefixer,
    logical_settings,
):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2018, month=11, day=1, tzinfo=timezone.utc)
    for _ in range(2):  # test pdb
        releases, _, _, _ = await mine_releases(
            ["src-d/go-git/alpha", "src-d/go-git/beta"],
            {},
            None,
            {},
            time_from,
            time_to,
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_match_setting_tag_logical,
            logical_settings,
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            rdb,
            None,
            with_avatars=False,
            with_pr_titles=False,
            with_deployments=False,
        )
        await wait_deferred()
        counts = {
            "github.com/src-d/go-git/alpha": 0,
            "github.com/src-d/go-git/beta": 0,
        }
        prs = counts.copy()
        for r, f in releases:
            counts[(repo := r[Release.repository_full_name.name])] += 1
            prs[repo] += len(f["prs_" + PullRequest.number.name])
        assert counts == {
            "github.com/src-d/go-git/alpha": 44,
            "github.com/src-d/go-git/beta": 28,
        }
        assert prs == {
            "github.com/src-d/go-git/alpha": 92,
            "github.com/src-d/go-git/beta": 58,
        }


@with_defer
async def test_mine_releases_logical_label(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag_logical,
    prefixer,
    logical_settings_labels,
):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=11, day=1, tzinfo=timezone.utc)
    for _ in range(2):  # test pdb
        releases, _, _, _ = await mine_releases(
            ["src-d/go-git/alpha", "src-d/go-git/beta"],
            {},
            None,
            {},
            time_from,
            time_to,
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_match_setting_tag_logical,
            logical_settings_labels,
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            rdb,
            None,
            with_avatars=False,
            with_pr_titles=False,
            with_deployments=False,
        )
        await wait_deferred()
        counts = {
            "github.com/src-d/go-git/alpha": 0,
            "github.com/src-d/go-git/beta": 0,
        }
        prs = counts.copy()
        deltas = counts.copy()
        commits = counts.copy()
        for r, f in releases:
            counts[(repo := r[Release.repository_full_name.name])] += 1
            prs[repo] += len(f["prs_" + PullRequest.number.name])
            deltas[repo] += f.additions + f.deletions
            commits[repo] += f.commits_count
        assert counts == {
            "github.com/src-d/go-git/alpha": 53,
            "github.com/src-d/go-git/beta": 37,
        }
        assert prs == {
            "github.com/src-d/go-git/alpha": 8,
            "github.com/src-d/go-git/beta": 5,
        }
        assert deltas == {
            "github.com/src-d/go-git/alpha": 5564,
            "github.com/src-d/go-git/beta": 271,
        }
        assert commits == {
            "github.com/src-d/go-git/alpha": 49,
            "github.com/src-d/go-git/beta": 6,
        }


@pytest.mark.parametrize("tag, count", [("v4.0.0", 7), ("v4.0.0~post1", 8)])
@with_defer
async def test_mine_releases_twins(
    mdb_rw,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    tag,
    count,
):
    time_from = datetime(year=2017, month=6, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2018, month=2, day=1, tzinfo=timezone.utc)
    await mdb_rw.execute(
        insert(Release).values(
            {
                Release.node_id: 100500,
                Release.acc_id: 6366825,
                Release.repository_full_name: "src-d/go-git",
                Release.repository_node_id: 40550,
                Release.author: "mcuadros",
                Release.author_node_id: 39789,
                Release.name: "v4.0.0",
                Release.published_at: datetime(2018, 1, 9, tzinfo=timezone.utc),
                Release.tag: tag,
                Release.url: "https://github.com/src-d/go-git/commit/tag/v4.0.0",
                Release.sha: "bf3b1f1fb9e0a04d0f87511a7ded2562b48a19d8",
                Release.commit_id: 2757510,
            },
        ),
    )
    try:
        releases, _, _, _ = await mine_releases(
            ["src-d/go-git"],
            {},
            None,
            {},
            time_from,
            time_to,
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            1,
            (6366825,),
            mdb_rw,
            pdb,
            rdb,
            None,
            with_deployments=False,
        )
        assert len(releases) == count
        passed = False
        for dikt, facts in releases:
            if dikt[Release.node_id.name] != 41518:
                continue
            passed = True
            assert len(facts.commit_authors)
            assert len(facts["prs_" + PullRequest.number.name])
        assert passed
    finally:
        await mdb_rw.execute(delete(Release).where(Release.node_id == 100500))


@with_defer
async def test_override_first_releases_smoke(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    branches,
    default_branches,
    metrics_calculator_factory,
    bots,
):
    time_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=12, day=1, tzinfo=timezone.utc)
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        True,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        False,
    )
    time_from = datetime(year=2017, month=6, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_deployments=False,
    )
    await wait_deferred()

    data = await pdb.fetch_val(
        select([GitHubReleaseFacts.data]).where(GitHubReleaseFacts.id == 41512),
    )
    facts = ReleaseFacts(data)
    checked = 0
    for key in ReleaseFacts.f:
        if key.startswith("prs_"):
            checked += 1
            if key not in ("prs_title", "prs_jira"):
                assert len(facts[key] if facts[key] is not None else []) > 0, key
    assert checked == 7
    rows = await pdb.fetch_all(
        select([GitHubDonePullRequestFacts.data]).where(
            GitHubDonePullRequestFacts.release_node_id == 41512,
        ),
    )
    assert len(rows) == 14
    for row in rows:
        assert not PullRequestFacts(row[0]).release_ignored

    ignored = await override_first_releases(
        releases, {}, release_match_setting_tag, 1, pdb, threshold_factor=0,
    )
    assert ignored == 1

    data = await pdb.fetch_val(
        select([GitHubReleaseFacts.data]).where(GitHubReleaseFacts.id == 41512),
    )
    facts = ReleaseFacts(data)
    checked = 0
    for key in ReleaseFacts.f:
        if key.startswith("prs_"):
            checked += 1
            assert facts[key] is None or len(facts[key]) == 0, key
    assert checked == 7
    rows = await pdb.fetch_all(
        select([GitHubDonePullRequestFacts.data]).where(
            GitHubDonePullRequestFacts.release_node_id == 41512,
        ),
    )
    assert len(rows) == 14
    for row in rows:
        facts = PullRequestFacts(row[0])
        assert facts.released is None
        assert facts.release_ignored


@pytest.mark.parametrize("settings_index", [0, 1])
@with_defer
async def test_precomputed_releases_low_level(
    mdb,
    pdb,
    rdb,
    branches,
    default_branches,
    prefixer,
    release_match_setting_tag,
    release_match_setting_branch,
    settings_index,
    release_loader,
):
    release_settings = [release_match_setting_branch, release_match_setting_tag][settings_index]
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_settings,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert_array_equal(
        releases[Release.author.name].values == "",
        releases[Release.author_node_id.name].isnull().values,
    )
    prels = await release_loader._fetch_precomputed_releases(
        {ReleaseMatch(settings_index): {["master", ".*"][settings_index]: ["src-d/go-git"]}},
        time_from,
        time_to,
        prefixer,
        1,
        pdb,
    )
    prels = prels[releases.columns]
    assert_frame_equal(releases, prels)


@with_defer
async def test_precomputed_releases_ambiguous(
    mdb,
    pdb,
    rdb,
    branches,
    default_branches,
    prefixer,
    release_match_setting_tag,
    release_match_setting_branch,
    release_loader,
):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases_tag, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    releases_branch, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    prels = await release_loader._fetch_precomputed_releases(
        {
            ReleaseMatch.tag: {".*": ["src-d/go-git"]},
            ReleaseMatch.branch: {"master": ["src-d/go-git"]},
        },
        time_from,
        time_to,
        prefixer,
        1,
        pdb,
    )
    prels = prels[releases_tag.columns]
    assert_frame_equal(releases_tag, prels)


async def test_precomputed_release_timespans(pdb, release_loader):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime.now(timezone.utc) - timedelta(days=100)
    mg1 = {ReleaseMatch.tag: {".*": ["src-d/go-git"]}}
    async with pdb.connection() as pdb_conn:
        async with pdb_conn.transaction():
            await release_loader._store_precomputed_release_match_spans(
                mg1, {"src-d/go-git": ReleaseMatch.tag}, time_from, time_to, 1, pdb_conn,
            )
            mg2 = {ReleaseMatch.branch: {"master": ["src-d/go-git"]}}
            await release_loader._store_precomputed_release_match_spans(
                mg2, {"src-d/go-git": ReleaseMatch.branch}, time_from, time_to, 1, pdb_conn,
            )
            await release_loader._store_precomputed_release_match_spans(
                mg1,
                {"src-d/go-git": ReleaseMatch.tag},
                time_from - timedelta(days=300),
                time_to + timedelta(days=200),
                1,
                pdb_conn,
            )
    spans = await release_loader.fetch_precomputed_release_match_spans({**mg1, **mg2}, 1, pdb)
    assert len(spans) == 1
    assert len(spans["src-d/go-git"]) == 2
    assert spans["src-d/go-git"][ReleaseMatch.tag][0] == time_from - timedelta(days=300)
    assert spans["src-d/go-git"][ReleaseMatch.tag][1] <= datetime.now(timezone.utc)
    assert spans["src-d/go-git"][ReleaseMatch.branch] == (time_from, time_to)


@with_defer
async def test_precomputed_releases_append(
    mdb,
    pdb,
    rdb,
    branches,
    default_branches,
    release_match_setting_tag,
    release_loader,
    prefixer,
):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases_tag1, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from + timedelta(days=300),
        time_to - timedelta(days=900),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert len(releases_tag1) == 39
    releases_tag2, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert len(releases_tag2) == 53


@with_defer
async def test_precomputed_releases_tags_after_branches(
    mdb,
    pdb,
    rdb,
    branches,
    default_branches,
    release_match_setting_branch,
    release_match_setting_tag_or_branch,
    release_loader,
    prefixer,
):
    # we don't have tags within our reach for this time interval
    time_from = datetime(year=2017, month=3, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=4, day=1, tzinfo=timezone.utc)
    releases_branch, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert len(releases_branch) == 15
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}

    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases_branch, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
        release_match_setting_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert len(releases_branch) == 772

    releases_tag, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
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
    await wait_deferred()
    assert len(releases_tag) == 53
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}

    releases_tag, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
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
    assert len(releases_tag) == 53
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@pytest.mark.flaky(reruns=3)
@with_defer
async def test_mine_releases_by_name(
    mdb,
    pdb,
    rdb,
    branches,
    default_branches,
    release_match_setting_branch,
    release_match_setting_tag_or_branch,
    prefixer,
    cache,
):
    # we don't have tags within our reach for this time interval
    time_from = datetime(year=2017, month=3, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=4, day=1, tzinfo=timezone.utc)
    releases, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_branch,
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
        with_deployments=False,
    )
    await wait_deferred()
    assert len(releases) == 15
    names = {"36c78b9d1b1eea682703fb1cbb0f4f3144354389", "v4.0.0"}
    releases, _, _ = await mine_releases_by_name(
        {"src-d/go-git": names},
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    assert len(releases) == 2
    for _, facts in releases:
        assert len(facts["prs_" + PullRequest.title.name]) == len(
            facts["prs_" + PullRequest.node_id.name],
        )
    releases_dict = {r[0][Release.name.name]: r for r in releases}
    assert releases_dict.keys() == names
    assert (
        len(
            releases_dict["36c78b9d1b1eea682703fb1cbb0f4f3144354389"][1][
                "prs_" + PullRequest.number.name
            ],
        )
        == 1
    )
    assert len(releases_dict["v4.0.0"][1]["prs_" + PullRequest.number.name]) == 62
    releases2, _, _ = await mine_releases_by_name(
        {"src-d/go-git": names},
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    assert str(releases) == str(releases2)


def _mk_rel_match_settings(
    *,
    branches: Optional[str] = None,
    tags: Optional[str] = None,
    events: Optional[str] = None,
) -> ReleaseMatchSetting:
    kwargs = {"branches": branches or "", "tags": tags or "", "events": events or ""}

    if branches is not None and tags is not None:
        kwargs["match"] = ReleaseMatch.tag_or_branch
    elif branches is not None:
        kwargs["match"] = ReleaseMatch.branch
    elif tags is not None:
        kwargs["match"] = ReleaseMatch.tag
    elif events is not None:
        kwargs["match"] = ReleaseMatch.event
    else:
        raise ValueError("At least one keywork argument is needed")
    return ReleaseMatchSetting(**kwargs)
