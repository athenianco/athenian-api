import pytest

from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.defer import wait_deferred, with_defer


async def test_extract_branches_zero(mdb):
    branches, defaults = await extract_branches(["src-d/gitbase"], mdb, None)
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}
    async with mdb.connection() as conn:
        branches, defaults = await extract_branches(["src-d/gitbase"], conn, None)
        assert branches.empty
        assert defaults == {"src-d/gitbase": "master"}


async def test_extract_branches_trash(mdb):
    branches, defaults = await extract_branches(["src-d/whatever"], mdb, None)
    assert branches.empty
    assert defaults == {"src-d/whatever": "master"}


@with_defer
async def test_extract_branches_cache(mdb, cache):
    branches, defaults = await extract_branches(["src-d/go-git"], mdb, cache)
    await wait_deferred()
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master"}
    branches, defaults = await extract_branches(["src-d/go-git"], None, cache)
    await wait_deferred()
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master"}
    with pytest.raises(AttributeError):
        await extract_branches(["src-d/go-git", "src-d/gitbase"], None, cache)
