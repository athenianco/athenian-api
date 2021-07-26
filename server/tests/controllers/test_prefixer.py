import pytest

from athenian.api.controllers.prefixer import Prefixer
from athenian.api.defer import wait_deferred, with_defer


@with_defer
async def test_prefixer_load(mdb, cache):
    for i in range(2):
        prefixer = await Prefixer.load((6366825,), mdb if i == 0 else None, cache)
        await wait_deferred()
        assert len(prefixer.user_node_to_prefixed_login) == \
               len(prefixer.user_login_to_prefixed_login) == 930
        assert "vmarkovtsev" in prefixer.user_login_to_prefixed_login
        assert len(prefixer.repo_node_to_prefixed_name) == 306
        assert len(prefixer.repo_name_to_prefixed_name) == 292
        assert "src-d/go-git" in prefixer.repo_name_to_prefixed_name


async def test_prefixer_schedule_load(mdb):
    prefixer = Prefixer.schedule_load((6366825,), mdb, None)
    prefixer = await prefixer.load()
    assert prefixer.user_login_to_prefixed_login
    promise = prefixer.as_promise()
    assert prefixer == await promise.load()


async def test_prefixer_schedule_cancel(mdb):
    prefixer = Prefixer.schedule_load((6366825,), mdb, None)
    prefixer.cancel()
    with pytest.raises(AssertionError):
        await prefixer.load()


async def test_prefixer_sequences(mdb):
    prefixer = await Prefixer.load((6366825,), mdb, None)
    assert prefixer.prefix_user_logins(["vmarkovtsev"]) == ["github.com/vmarkovtsev"]
    assert prefixer.prefix_repo_names(["src-d/go-git"]) == ["github.com/src-d/go-git"]
    assert prefixer.resolve_user_nodes(["MDQ6VXNlcjI3OTM1NTE="]) == ["github.com/vmarkovtsev"]
    assert prefixer.resolve_repo_nodes(["MDEwOlJlcG9zaXRvcnk0NDczOTA0NA=="]) == \
           ["github.com/src-d/go-git"]
