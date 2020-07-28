import pytest

from athenian.api.controllers.miners.github.branches import extract_branches


async def test_extract_branches_zero(mdb):
    branches, defaults = await extract_branches(["src-d/gitbase"], mdb, None)
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}
    async with mdb.connection() as conn:
        branches, defaults = await extract_branches(["src-d/gitbase"], conn, None)
        assert branches.empty
        assert defaults == {"src-d/gitbase": "master"}


async def test_extract_branches_cache(mdb, cache):
    branches, defaults = await extract_branches(["src-d/go-git"], mdb, cache)
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master"}
    branches, defaults = await extract_branches(["src-d/go-git"], None, cache)
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master"}
    with pytest.raises(AttributeError):
        await extract_branches(["src-d/go-git", "src-d/gitbase"], None, cache)
