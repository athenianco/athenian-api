from athenian.api.controllers.miners.github.user import mine_user_avatars, UserAvatarKeys
from athenian.api.defer import wait_deferred, with_defer


@with_defer
async def test_mine_user_avatars_cache(mdb, cache):
    avatars = await mine_user_avatars(
        ["vmarkovtsev", "mcuadros"], UserAvatarKeys.LOGIN, (6366825,), mdb, cache)
    await wait_deferred()
    assert avatars == [
        ("mcuadros", "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        ("vmarkovtsev", "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
    avatars = await mine_user_avatars(
        ["vmarkovtsev", "mcuadros"], UserAvatarKeys.PREFIXED_LOGIN, (6366825,), None, cache)
    await wait_deferred()
    assert avatars == [
        ("github.com/mcuadros", "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        ("github.com/vmarkovtsev", "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
    avatars = await mine_user_avatars(
        ["vmarkovtsev", "mcuadros"], UserAvatarKeys.NODE, (6366825,), None, cache)
    assert avatars == [
        (39789, "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"),
        (40020, "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4"),
    ]
