from operator import itemgetter

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.contributors import mine_contributors
from athenian.api.internal.settings import LogicalRepositorySettings
from tests.conftest import has_memcached


@with_defer
async def test_mine_contributors_expected_cache_miss_with_stats(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    memcached,
    prefixer,
):
    if has_memcached:
        cache = memcached

    contribs_with_stats = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        True,
        [],
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
    contribs_with_no_stats = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        False,
        [],
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

    assert len(contribs_with_stats) == len(contribs_with_no_stats)
    _assert_contribs_equal(contribs_with_stats, contribs_with_no_stats, [True, False])


@with_defer
async def test_mine_contributors_expected_cache_miss_with_different_roles(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    memcached,
    prefixer,
):
    if has_memcached:
        cache = memcached

    authors = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        True,
        ["author"],
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
    mergers = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        True,
        ["merger"],
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

    assert len(authors) == 172
    assert len(mergers) == 8


@with_defer
async def test_mine_contributors_with_empty_and_all_roles(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
):
    contribs_with_empty_roles = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        True,
        [],
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
    contribs_with_all_roles = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        True,
        [
            "author",
            "reviewer",
            "commit_author",
            "commit_committer",
            "commenter",
            "merger",
            "releaser",
            "member",
        ],
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

    assert len(contribs_with_empty_roles) == len(contribs_with_all_roles)
    _assert_contribs_equal(contribs_with_empty_roles, contribs_with_all_roles, [True, True])


@with_defer
async def test_mine_contributors_user_roles(mdb, pdb, rdb, release_match_setting_tag, prefixer):
    authors_with_stats = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        True,
        ["author"],
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
    authors_with_no_stats = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        False,
        ["author"],
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

    assert len(authors_with_stats) == 172
    assert len(authors_with_no_stats) == 172
    _assert_contribs_equal(authors_with_stats, authors_with_no_stats, [True, False])

    mergers_with_stats = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        True,
        ["merger"],
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
    mergers_with_no_stats = await mine_contributors(
        ["src-d/go-git"],
        None,
        None,
        False,
        ["merger"],
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

    actual_merges_count = {c["login"]: c["stats"]["merger"] for c in mergers_with_stats}
    expected_merges_count = {
        "alcortesm": 11,
        "erizocosmico": 3,
        "mcuadros": 489,
        "strib": 2,
        "ajnavarro": 3,
        "orirawlings": 9,
        "jfontan": 3,
        "smola": 34,
    }

    assert actual_merges_count == expected_merges_count
    _assert_contribs_equal(mergers_with_stats, mergers_with_no_stats, [True, False])


def _assert_contribs_equal(contribs_1, contribs_2, with_stats):
    for c1, c2 in zip(
        sorted(contribs_1, key=itemgetter("login")), sorted(contribs_2, key=itemgetter("login")),
    ):
        assert ("stats" in c1) == with_stats[0]
        assert ("stats" in c2) == with_stats[1]

        if not all(with_stats):
            c1.pop("stats", None)
            c2.pop("stats", None)

        assert c1 == c2
