from operator import itemgetter

from athenian.api.controllers.miners.github.contributors import \
    mine_contributors
from athenian.api.defer import wait_deferred, with_defer
from tests.conftest import has_memcached


@with_defer
async def test_mine_contributors_expected_cache_miss_with_stats(mdb, cache, memcached):
    if has_memcached:
        cache = memcached

    contribs_with_stats = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, cache, with_stats=True)
    await wait_deferred()
    contribs_with_no_stats = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, cache, with_stats=False)

    assert len(contribs_with_stats) == len(contribs_with_no_stats)
    _assert_contribs_equal(contribs_with_stats, contribs_with_no_stats, [True, False])


@with_defer
async def test_mine_contributors_expected_cache_miss_with_different_roles(mdb, cache, memcached):
    if has_memcached:
        cache = memcached

    authors = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, cache, with_stats=True, as_roles=["author"])
    await wait_deferred()
    mergers = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, cache, with_stats=True, as_roles=["merger"])

    assert len(authors) == 172
    assert len(mergers) == 8


async def test_mine_contributors_with_empty_and_all_roles(mdb):
    contribs_with_no_roles = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, None, as_roles=[])
    contribs_with_empty_roles = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, None)
    contribs_with_all_roles = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, None,
        as_roles=["author", "reviewer", "commit_author", "commit_committer",
                  "commenter", "merger", "releaser"])

    assert (
        len(contribs_with_no_roles) ==
        len(contribs_with_empty_roles) ==
        len(contribs_with_all_roles)
    )
    _assert_contribs_equal(contribs_with_no_roles, contribs_with_all_roles, [True, True])
    _assert_contribs_equal(contribs_with_empty_roles, contribs_with_all_roles, [True, True])


async def test_mine_contributors_as_roles(mdb):
    authors_with_stats = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, None, with_stats=True, as_roles=["author"])
    authors_with_no_stats = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, None, with_stats=False, as_roles=["author"])

    assert len(authors_with_stats) == 172
    assert len(authors_with_no_stats) == 172
    _assert_contribs_equal(authors_with_stats, authors_with_no_stats, [True, False])

    mergers_with_stats = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, None, with_stats=True, as_roles=["merger"])
    mergers_with_no_stats = await mine_contributors(
        ["src-d/go-git"], None, None, mdb, None, with_stats=False, as_roles=["merger"])

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
