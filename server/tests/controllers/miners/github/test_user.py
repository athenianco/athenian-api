from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.user import UserAvatarKeys, mine_user_avatars


@with_defer
async def test_mine_user_avatars_cache_logins(mdb, cache):
    avatars = await mine_user_avatars(
        UserAvatarKeys.LOGIN, (6366825,), mdb, cache, logins=["vmarkovtsev", "mcuadros"],
    )
    await wait_deferred()
    assert avatars == [
        ("mcuadros", "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        ("vmarkovtsev", "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
    avatars = await mine_user_avatars(
        UserAvatarKeys.PREFIXED_LOGIN, (6366825,), None, cache, logins=["vmarkovtsev", "mcuadros"],
    )
    await wait_deferred()
    assert avatars == [
        ("github.com/mcuadros", "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        ("github.com/vmarkovtsev", "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
    avatars = await mine_user_avatars(
        UserAvatarKeys.NODE, (6366825,), None, cache, logins=["vmarkovtsev", "mcuadros"],
    )
    assert avatars == [
        (39789, "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        (40020, "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]


@with_defer
async def test_mine_user_avatars_cache_nodes(mdb, cache):
    avatars = await mine_user_avatars(
        UserAvatarKeys.LOGIN, (6366825,), mdb, cache, nodes=[40020, 39789],
    )
    await wait_deferred()
    assert avatars == [
        ("mcuadros", "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        ("vmarkovtsev", "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
    avatars = await mine_user_avatars(
        UserAvatarKeys.PREFIXED_LOGIN, (6366825,), None, cache, nodes=[40020, 39789],
    )
    await wait_deferred()
    assert avatars == [
        ("github.com/mcuadros", "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        ("github.com/vmarkovtsev", "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
    avatars = await mine_user_avatars(
        UserAvatarKeys.NODE, (6366825,), None, cache, nodes=[40020, 39789],
    )
    assert avatars == [
        (39789, "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        (40020, "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
