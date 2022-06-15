from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.prefixer import Prefixer


@with_defer
async def test_prefixer_load(mdb, cache):
    for i in range(2):
        prefixer = await Prefixer.load((6366825,), mdb if i == 0 else None, cache)
        await wait_deferred()
        assert (
            len(prefixer.user_node_to_prefixed_login)
            == len(prefixer.user_login_to_prefixed_login)
            == 930
        )
        assert "vmarkovtsev" in prefixer.user_login_to_prefixed_login
        assert len(prefixer.repo_node_to_prefixed_name) == 306
        assert len(prefixer.repo_name_to_prefixed_name) == 292
        assert "src-d/go-git" in prefixer.repo_name_to_prefixed_name


async def test_prefixer_sequences(mdb):
    prefixer = await Prefixer.load((6366825,), mdb, None)
    assert prefixer.prefix_user_logins(["vmarkovtsev"]) == ["github.com/vmarkovtsev"]
    assert prefixer.prefix_repo_names(["src-d/go-git"]) == ["github.com/src-d/go-git"]
    assert prefixer.resolve_user_nodes([40020]) == ["github.com/vmarkovtsev"]
    assert prefixer.resolve_repo_nodes([40550]) == ["github.com/src-d/go-git"]
