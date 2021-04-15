import pytest

from athenian.api.controllers.prefixer import Prefixer


async def test_prefixer_load(mdb):
    prefixer = await Prefixer.load((6366825,), mdb)
    assert len(prefixer.user_node_map) == len(prefixer.user_login_map) == 929
    assert "vmarkovtsev" in prefixer.user_login_map
    assert len(prefixer.repo_node_map) == 306
    assert len(prefixer.repo_name_map) == 292
    assert "src-d/go-git" in prefixer.repo_name_map


async def test_prefixer_schedule_load(mdb):
    prefixer = Prefixer.schedule_load((6366825,), mdb)
    prefixer = await prefixer.load()
    assert prefixer.user_login_map
    promise = prefixer.as_promise()
    assert prefixer == await promise.load()


async def test_prefixer_schedule_cancel(mdb):
    prefixer = Prefixer.schedule_load((6366825,), mdb)
    prefixer.cancel()
    with pytest.raises(AssertionError):
        await prefixer.load()


async def test_prefixer_sequences(mdb):
    prefixer = await Prefixer.load((6366825,), mdb)
    assert prefixer.prefix_user_logins(["vmarkovtsev"]) == ["github.com/vmarkovtsev"]
    assert prefixer.prefix_repo_names(["src-d/go-git"]) == ["github.com/src-d/go-git"]
    assert prefixer.resolve_user_nodes(["MDQ6VXNlcjI3OTM1NTE="]) == ["github.com/vmarkovtsev"]
    assert prefixer.resolve_repo_nodes(["MDEwOlJlcG9zaXRvcnk0NDczOTA0NA=="]) == \
           ["github.com/src-d/go-git"]
