from datetime import datetime, timedelta, timezone
from sqlite3 import OperationalError

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import delete, func, select, sql
from sqlalchemy.schema import CreateTable

from athenian.api.async_utils import read_sql_query
from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.precomputed_prs import store_precomputed_done_facts
from athenian.api.internal.miners.github.pull_request import PullRequestFactsMiner
from athenian.api.internal.miners.github.release_match import (
    PullRequestToReleaseMapper,
    ReleaseToPullRequestMapper,
)
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
)
from athenian.api.models.metadata.github import NodeCommit, PullRequest, PullRequestLabel, Release
from athenian.api.models.precomputed.models import GitHubCommitHistory
from tests.conftest import _metadata_db
from tests.controllers.conftest import SAMPLE_BOTS
from tests.controllers.test_filter_controller import force_push_dropped_go_git_pr_numbers


@with_defer
async def test_map_prs_to_releases_cache(
    branches,
    default_branches,
    dag,
    mdb,
    pdb,
    rdb,
    cache,
    release_loader,
    prefixer,
):
    prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.number == 1126),
        mdb,
        PullRequest,
        index=[PullRequest.node_id.name, PullRequest.repository_full_name.name],
    )
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    release_settings = _generate_repo_settings(prs)
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
        None,
    )
    tag = "https://github.com/src-d/go-git/releases/tag/v4.12.0"
    for i in range(2):
        released_prs, facts, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
            prs,
            releases,
            matched_bys,
            branches,
            default_branches,
            time_to,
            dag,
            release_settings,
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            cache,
        )
        await wait_deferred()
        assert isinstance(facts, dict)
        assert len(facts) == 0
        assert len(cache.mem) > 0
        assert len(released_prs) == 1, str(i)
        assert released_prs.iloc[0][Release.url.name] == tag
        assert released_prs.iloc[0][Release.published_at.name] == pd.Timestamp(
            "2019-06-18 22:57:34+0000", tzinfo=timezone.utc,
        )
        assert released_prs.iloc[0][Release.author.name] == "mcuadros"
    released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        prs,
        releases,
        matched_bys,
        branches,
        default_branches,
        time_to,
        dag,
        release_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    # the PR was merged and released in the past, we must detect that
    assert len(released_prs) == 1
    assert released_prs.iloc[0][Release.url.name] == tag


@with_defer
async def test_map_prs_to_releases_pdb(
    branches,
    default_branches,
    dag,
    mdb,
    pdb,
    rdb,
    release_loader,
    prefixer,
):
    prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.number.in_((1126, 1180))),
        mdb,
        PullRequest,
        index=[PullRequest.node_id.name, PullRequest.repository_full_name.name],
    )
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    release_settings = _generate_repo_settings(prs)
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
        None,
    )
    released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        prs,
        releases,
        matched_bys,
        branches,
        default_branches,
        time_to,
        dag,
        release_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    await wait_deferred()
    assert len(released_prs) == 1
    dummy_mdb = await Database("sqlite://").connect()
    try:
        prlt = PullRequestLabel.__table__
        if prlt.schema:
            for table in (PullRequestLabel, NodeCommit):
                table = table.__table__
                table.name = "%s.%s" % (table.schema, table.name)
                table.schema = None
        for table in (PullRequestLabel, NodeCommit):
            await dummy_mdb.execute(CreateTable(table.__table__))
        released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
            prs,
            releases,
            matched_bys,
            branches,
            default_branches,
            time_to,
            dag,
            release_settings,
            prefixer,
            1,
            (6366825,),
            dummy_mdb,
            pdb,
            None,
        )
        assert len(released_prs) == 1
    finally:
        if "." in prlt.name:
            for table in (PullRequestLabel, NodeCommit):
                table = table.__table__
                table.schema, table.name = table.name.split(".")
        await dummy_mdb.disconnect()


@with_defer
async def test_map_prs_to_releases_empty(
    branches,
    default_branches,
    dag,
    mdb,
    pdb,
    rdb,
    cache,
    release_loader,
    prefixer,
):
    prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.number == 1231),
        mdb,
        PullRequest,
        index=[PullRequest.node_id.name, PullRequest.repository_full_name.name],
    )
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    release_settings = _generate_repo_settings(prs)
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
        None,
    )
    for i in range(2):
        released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
            prs,
            releases,
            matched_bys,
            branches,
            default_branches,
            time_to,
            dag,
            release_settings,
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            cache,
        )
        assert len(cache.mem) == 1, i
        assert released_prs.empty
    prs = prs.iloc[:0]
    released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        prs,
        releases,
        matched_bys,
        branches,
        default_branches,
        time_to,
        dag,
        release_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        cache,
    )
    assert len(cache.mem) == 1
    assert released_prs.empty


@with_defer
async def test_map_prs_to_releases_precomputed_released(
    branches,
    default_branches,
    dag,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    release_loader,
    pr_miner,
    prefixer,
):
    time_to = datetime(year=2019, month=8, day=2, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=2)

    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches,
        default_branches,
        False,
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
    facts_miner = PullRequestFactsMiner(SAMPLE_BOTS)
    true_prs = [pr for pr in miner if pr.release[Release.published_at.name] is not None]
    facts = [facts_miner(pr) for pr in true_prs]
    prs = pd.DataFrame([pr.pr for pr in true_prs]).set_index(
        [PullRequest.node_id.name, PullRequest.repository_full_name.name],
    )
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

    await pdb.execute(delete(GitHubCommitHistory))
    dummy_mdb = await Database("sqlite://").connect()
    prlt = PullRequestLabel.__table__
    try:
        if prlt.schema:
            for table in (PullRequestLabel, NodeCommit):
                table = table.__table__
                table.name = "%s.%s" % (table.schema, table.name)
                table.schema = None
        for table in (PullRequestLabel, NodeCommit):
            await dummy_mdb.execute(CreateTable(table.__table__))

        await store_precomputed_done_facts(
            true_prs,
            facts,
            datetime(2022, 1, 1, tzinfo=timezone.utc),
            default_branches,
            release_match_setting_tag,
            1,
            pdb,
        )

        released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
            prs,
            releases,
            matched_bys,
            branches,
            default_branches,
            time_to,
            dag,
            release_match_setting_tag,
            prefixer,
            1,
            (6366825,),
            dummy_mdb,
            pdb,
            None,
        )
        assert len(released_prs) == len(prs)
    finally:
        if "." in prlt.name:
            for table in (PullRequestLabel, NodeCommit):
                table = table.__table__
                table.schema, table.name = table.name.split(".")
        await dummy_mdb.disconnect()


@pytest.mark.flaky(reruns=2)
@with_defer
async def test_map_prs_to_releases_smoke_metrics(
    branches,
    default_branches,
    dag,
    mdb,
    pdb,
    rdb,
    release_loader,
    worker_id,
    prefixer,
):
    try:
        await mdb.fetch_val(select([func.count(PullRequestLabel.node_id)]))
    except OperationalError as e:
        # this happens sometimes, we have to reset the DB and proceed to the second lap
        await mdb.disconnect()
        _metadata_db(worker_id, True)
        raise e from None
    time_from = datetime(year=2015, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    filters = [
        sql.and_(PullRequest.merged_at > time_from, PullRequest.created_at < time_to),
        PullRequest.repository_full_name.in_(["src-d/go-git"]),
        PullRequest.user_login.in_(["mcuadros", "vmarkovtsev"]),
    ]
    prs = await read_sql_query(
        select([PullRequest]).where(sql.and_(*filters)),
        mdb,
        PullRequest,
        index=[PullRequest.node_id.name, PullRequest.repository_full_name.name],
    )
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    release_settings = _generate_repo_settings(prs)
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
        None,
    )
    released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        prs,
        releases,
        matched_bys,
        branches,
        default_branches,
        time_to,
        dag,
        release_settings,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
    )
    assert set(released_prs[Release.url.name].unique()) == {
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc10",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc11",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc13",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc12",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc14",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc15",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0",
        "https://github.com/src-d/go-git/releases/tag/v4.2.0",
        "https://github.com/src-d/go-git/releases/tag/v4.1.1",
        "https://github.com/src-d/go-git/releases/tag/v4.2.1",
        "https://github.com/src-d/go-git/releases/tag/v4.5.0",
        "https://github.com/src-d/go-git/releases/tag/v4.11.0",
        "https://github.com/src-d/go-git/releases/tag/v4.7.1",
        "https://github.com/src-d/go-git/releases/tag/v4.8.0",
        "https://github.com/src-d/go-git/releases/tag/v4.10.0",
        "https://github.com/src-d/go-git/releases/tag/v4.12.0",
        "https://github.com/src-d/go-git/releases/tag/v4.13.0",
    }


@with_defer
async def test__find_dead_merged_prs_smoke(mdb):
    prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.merged_at.isnot(None)),
        mdb,
        PullRequest,
        index=[PullRequest.node_id.name, PullRequest.repository_full_name.name],
    )
    prs["dead"] = False
    prs.loc[prs[PullRequest.number.name].isin(force_push_dropped_go_git_pr_numbers), "dead"] = True
    dead_prs = await PullRequestToReleaseMapper._find_dead_merged_prs(prs)
    assert len(dead_prs) == len(force_push_dropped_go_git_pr_numbers)
    assert dead_prs[Release.published_at.name].isnull().all()
    assert (dead_prs[matched_by_column] == ReleaseMatch.force_push_drop).all()
    dead_prs = await mdb.fetch_all(
        select([PullRequest.number]).where(
            PullRequest.node_id.in_(dead_prs.index.get_level_values(0).values),
        ),
    )
    assert {pr[0] for pr in dead_prs} == set(force_push_dropped_go_git_pr_numbers)


@with_defer
async def test__extract_released_commits_4_0_0(
    release_loader,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    dag,
):
    time_from = datetime(year=2016, month=1, day=8, tzinfo=timezone.utc)
    time_to = datetime(year=2018, month=1, day=9, tzinfo=timezone.utc)
    time_boundary = datetime(year=2018, month=1, day=7, tzinfo=timezone.utc)
    dag = dag["src-d/go-git"][1]
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
    assert len(releases) == 29
    hashes1 = ReleaseToPullRequestMapper._extract_released_commits(releases, dag, time_boundary)
    releases = releases.iloc[:2]
    hashes2 = ReleaseToPullRequestMapper._extract_released_commits(releases, dag, time_boundary)
    assert_array_equal(hashes1, hashes2)
    assert len(hashes1) == 181


"""
https://athenianco.atlassian.net/browse/DEV-250

async def test_map_prs_to_releases_miguel(
        mdb, pdb, rdb, release_match_setting_tag, cache, release_loader, prefixer):
    miguel_pr = await read_sql_query(
        select([PullRequest]).where(PullRequest.number == 907),
        mdb, PullRequest, index=[PullRequest.node_id.name, PullRequest.repository_full_name.name])
    # https://github.com/src-d/go-git/pull/907
    assert len(miguel_pr) == 1
    time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 5, 1, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, None, time_from, time_to,
        release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, cache)
    released_prs, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        miguel_pr, releases, matched_bys, pd.DataFrame(), {}, time_to,
        release_match_setting_tag, prefixer, 1, (6366825,), mdb, pdb, cache)
    assert len(released_prs) == 1
"""


def _generate_repo_settings(prs: pd.DataFrame) -> ReleaseSettings:
    return ReleaseSettings(
        {
            "github.com/"
            + r: ReleaseMatchSetting(branches="", tags=".*", events=".*", match=ReleaseMatch.tag)
            for r in prs.index.get_level_values(1).values
        },
    )


@with_defer
async def test_map_releases_to_prs_early_merges(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    releases_to_prs_mapper,
    prefixer,
):
    prs, releases, _, matched_bys, dag, _ = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2018, month=1, day=7, tzinfo=timezone.utc),
        datetime(year=2018, month=1, day=9, tzinfo=timezone.utc),
        [],
        [],
        JIRAFilter.empty(),
        release_match_setting_tag,
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
    assert len(releases) == 1
    assert len(prs) == 61
    assert (
        prs[PullRequest.merged_at.name] > datetime(year=2017, month=9, day=4, tzinfo=timezone.utc)
    ).all()
    assert isinstance(dag, dict)
    dag = dag["src-d/go-git"][1]
    assert len(dag) == 3
    assert len(dag[0]) == 1012
    assert dag[0].dtype == np.dtype("S40")
    assert len(dag[1]) == 1013
    assert dag[1].dtype == np.uint32
    assert len(dag[2]) == dag[1][-1]
    assert dag[2].dtype == np.uint32


@with_defer
async def test_map_releases_to_prs_smoke(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag_or_branch,
    releases_to_prs_mapper,
    prefixer,
):
    for _ in range(2):
        (
            prs,
            releases,
            new_settings,
            matched_bys,
            dag,
            _,
        ) = await releases_to_prs_mapper.map_releases_to_prs(
            ["src-d/go-git"],
            branches,
            default_branches,
            datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
            datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
            [],
            [],
            JIRAFilter.empty(),
            release_match_setting_tag_or_branch,
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
            cache,
        )
        await wait_deferred()
        assert len(prs) == 7
        assert len(dag["src-d/go-git"][1][0]) == 1508
        assert (
            prs[PullRequest.merged_at.name]
            < pd.Timestamp("2019-07-31 00:00:00", tzinfo=timezone.utc)
        ).all()
        assert (
            prs[PullRequest.merged_at.name]
            > pd.Timestamp("2019-06-19 00:00:00", tzinfo=timezone.utc)
        ).all()
        assert len(releases) == 2
        assert set(releases[Release.sha.name]) == {
            b"0d1a009cbb604db18be960db5f1525b99a55d727",
            b"6241d0e70427cb0db4ca00182717af88f638268c",
        }
        assert new_settings == ReleaseSettings(
            {
                "github.com/src-d/go-git": ReleaseMatchSetting(
                    branches="master", tags=".*", events=".*", match=ReleaseMatch.tag,
                ),
            },
        )
        assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@with_defer
async def test_map_releases_to_prs_no_truncate(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    releases_to_prs_mapper,
    prefixer,
):
    prs, releases, _, matched_bys, _, _ = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2018, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2018, month=12, day=2, tzinfo=timezone.utc),
        [],
        [],
        JIRAFilter.empty(),
        release_match_setting_tag,
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
        truncate=False,
    )
    assert len(prs) == 8
    assert len(releases) == 5 + 7
    assert releases[Release.published_at.name].is_monotonic_decreasing
    assert releases.index.is_monotonic
    assert "v4.13.1" in releases[Release.tag.name].values


@with_defer
async def test_map_releases_to_prs_empty(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag,
    releases_to_prs_mapper,
    prefixer,
):
    prs, releases, _, matched_bys, _, _ = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2019, month=7, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [],
        [],
        JIRAFilter.empty(),
        release_match_setting_tag,
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
        cache,
    )
    await wait_deferred()
    assert prs.empty
    assert len(cache.mem) == 5
    assert len(releases) == 2
    assert set(releases[Release.sha.name]) == {
        b"0d1a009cbb604db18be960db5f1525b99a55d727",
        b"6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    prs, releases, _, matched_bys, _, _ = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2019, month=7, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [],
        [],
        JIRAFilter.empty(),
        ReleaseSettings(
            {
                "github.com/src-d/go-git": ReleaseMatchSetting(
                    branches="master", tags=".*", events=".*", match=ReleaseMatch.branch,
                ),
            },
        ),
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
        cache,
    )
    assert prs.empty
    assert len(cache.mem) == 12
    assert len(releases) == 19
    assert releases.iloc[0].to_dict() == {
        "node_id": 2755389,
        "repository_node_id": 40550,
        "published_at": pd.Timestamp("2019-11-01 09:08:16+0000", tz="UTC"),
        "sha": b"1a7db85bca7027d90afdb5ce711622aaac9feaed",
        "commit_id": 2755389,
        "repository_full_name": "src-d/go-git",
        "author_node_id": 39789,
        "name": "1a7db85bca7027d90afdb5ce711622aaac9feaed",
        "tag": None,
        "url": "https://github.com/src-d/go-git/commit/1a7db85bca7027d90afdb5ce711622aaac9feaed",
        "matched_by": 0,
        "author": "mcuadros",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_map_releases_to_prs_blacklist(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag,
    releases_to_prs_mapper,
    prefixer,
):
    prs, releases, _, matched_bys, _, _ = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [],
        [],
        JIRAFilter.empty(),
        release_match_setting_tag,
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
        cache,
        pr_blacklist=PullRequest.node_id.notin_(
            [
                163378,
                163380,
                163395,
                163375,
                163377,
                163397,
                163396,
            ],
        ),
    )
    assert prs.empty
    assert len(releases) == 2
    assert set(releases[Release.sha.name]) == {
        b"0d1a009cbb604db18be960db5f1525b99a55d727",
        b"6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@pytest.mark.parametrize(
    "authors, mergers, n",
    [(["mcuadros"], [], 2), ([], ["mcuadros"], 7), (["mcuadros"], ["mcuadros"], 7)],
)
@with_defer
async def test_map_releases_to_prs_authors_mergers(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    prefixer,
    release_match_setting_tag,
    authors,
    mergers,
    n,
    releases_to_prs_mapper,
):
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
        datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        authors,
        mergers,
        JIRAFilter.empty(),
        release_match_setting_tag,
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
        cache,
    )
    assert len(prs) == n
    assert len(releases) == 2
    assert set(releases[Release.sha.name]) == {
        b"0d1a009cbb604db18be960db5f1525b99a55d727",
        b"6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert new_settings == release_match_setting_tag
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@with_defer
async def test_map_releases_to_prs_hard(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag,
    releases_to_prs_mapper,
    prefixer,
):
    prs, releases, _, matched_bys, _, _ = await releases_to_prs_mapper.map_releases_to_prs(
        ["src-d/go-git"],
        branches,
        default_branches,
        datetime(year=2019, month=6, day=18, tzinfo=timezone.utc),
        datetime(year=2019, month=6, day=30, tzinfo=timezone.utc),
        [],
        [],
        JIRAFilter.empty(),
        release_match_setting_tag,
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
        cache,
    )
    assert len(prs) == 24
    assert len(releases) == 1
    assert set(releases[Release.sha.name]) == {
        b"f9a30199e7083bdda8adad3a4fa2ec42d25c1fdb",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@with_defer
async def test_map_releases_to_prs_future(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
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
        truncate=False,
    )
    assert len(prs) == 8
    assert releases is not None
    assert len(releases) == 12


@with_defer
async def test_map_releases_to_prs_precomputed_observed(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    releases_to_prs_mapper,
    prefixer,
):
    args = [
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
    ]
    prs1, _, _, _, _, precomputed_observed = await releases_to_prs_mapper.map_releases_to_prs(
        *args, truncate=False,
    )
    prs2 = await releases_to_prs_mapper.map_releases_to_prs(
        *args, truncate=False, precomputed_observed=precomputed_observed,
    )
    assert_frame_equal(prs1, prs2)
