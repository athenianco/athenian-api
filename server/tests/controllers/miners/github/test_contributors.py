from operator import itemgetter

import pytest

from athenian.api.controllers.miners.github.contributors import mine_contributors
from tests.conftest import has_memcached


async def test_mine_contributors_expected_cache_miss(mdb, cache, memcached):
    if not has_memcached:
        raise pytest.skip("no memcached")
    cache = memcached

    contribs_with_stats = await mine_contributors(["src-d/go-git"], mdb, cache=cache)
    contribs_with_no_stats = await mine_contributors(["src-d/go-git"], mdb, with_stats=False,
                                                     cache=cache)

    assert len(contribs_with_stats) == len(contribs_with_no_stats)

    for c_with_stats, c_no_stats in zip(
            sorted(contribs_with_stats, key=itemgetter("login")),
            sorted(contribs_with_no_stats, key=itemgetter("login"))):
        assert "stats" in c_with_stats
        assert "stats" not in c_no_stats

        c_with_stats.pop("stats")

        assert c_with_stats == c_no_stats
