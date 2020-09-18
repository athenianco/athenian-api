from operator import itemgetter

from athenian.api.controllers.miners.github.contributors import \
    mine_contributors
from athenian.api.defer import wait_deferred, with_defer
from tests.conftest import has_memcached


@with_defer
async def test_mine_contributors_expected_cache_miss_with_stats(
        mdb, pdb, release_match_setting_tag, cache, memcached):
    if has_memcached:
        cache = memcached

    contribs_with_stats = await mine_contributors(
        ["src-d/go-git"], None, None, True, [], release_match_setting_tag, mdb, pdb, cache)
    await wait_deferred()
    contribs_with_no_stats = await mine_contributors(
        ["src-d/go-git"], None, None, False, [], release_match_setting_tag, mdb, pdb, cache)

    assert len(contribs_with_stats) == len(contribs_with_no_stats)
    _assert_contribs_equal(contribs_with_stats, contribs_with_no_stats, [True, False])


@with_defer
async def test_mine_contributors_expected_cache_miss_with_different_roles(
        mdb, pdb, release_match_setting_tag, cache, memcached):
    if has_memcached:
        cache = memcached

    authors = await mine_contributors(
        ["src-d/go-git"], None, None, True, ["author"], release_match_setting_tag, mdb, pdb, cache)
    await wait_deferred()
    mergers = await mine_contributors(
        ["src-d/go-git"], None, None, True, ["merger"], release_match_setting_tag, mdb, pdb, cache)

    assert len(authors) == 172
    assert len(mergers) == 8


@with_defer
async def test_mine_contributors_with_empty_and_all_roles(mdb, pdb, release_match_setting_tag):
    contribs_with_empty_roles = await mine_contributors(
        ["src-d/go-git"], None, None, True, [], release_match_setting_tag, mdb, pdb, None)
    contribs_with_all_roles = await mine_contributors(
        ["src-d/go-git"], None, None, True,
        ["author", "reviewer", "commit_author", "commit_committer",
         "commenter", "merger", "releaser"],
        release_match_setting_tag, mdb, pdb, None)

    assert (
        len(contribs_with_empty_roles) ==
        len(contribs_with_all_roles)
    )
    _assert_contribs_equal(contribs_with_empty_roles, contribs_with_all_roles, [True, True])


@with_defer
async def test_mine_contributors_user_roles(mdb, pdb, release_match_setting_tag):
    authors_with_stats = await mine_contributors(
        ["src-d/go-git"], None, None, True, ["author"], release_match_setting_tag, mdb, pdb, None)
    authors_with_no_stats = await mine_contributors(
        ["src-d/go-git"], None, None, False, ["author"], release_match_setting_tag, mdb, pdb, None)

    assert len(authors_with_stats) == 172
    assert len(authors_with_no_stats) == 172
    _assert_contribs_equal(authors_with_stats, authors_with_no_stats, [True, False])

    mergers_with_stats = await mine_contributors(
        ["src-d/go-git"], None, None, True, ["merger"], release_match_setting_tag, mdb, pdb, None)
    mergers_with_no_stats = await mine_contributors(
        ["src-d/go-git"], None, None, False, ["merger"], release_match_setting_tag, mdb, pdb, None)

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
    for c1, c2 in zip(sorted(contribs_1, key=itemgetter("login")),
                      sorted(contribs_2, key=itemgetter("login"))):
        assert ("stats" in c1) == with_stats[0]
        assert ("stats" in c2) == with_stats[1]

        if not all(with_stats):
            c1.pop("stats", None)
            c2.pop("stats", None)

        assert c1 == c2
