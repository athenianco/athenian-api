import dataclasses
from typing import Any

import pytest

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.prefixer import Prefixer, RepositoryName, RepositoryReference
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID


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


class TestRepositoryName:
    def test_from_prefixed_wrong_value(self) -> None:
        for bad_value in ("org/repo", "org/repo/logic"):
            with pytest.raises(ValueError):
                RepositoryName.from_prefixed(bad_value)

    def test_from_prefixed(self) -> None:
        name = RepositoryName.from_prefixed("github.com/org/repo")
        assert name.prefix == "github.com"
        assert name.owner == "org"
        assert name.physical == "repo"
        assert name.logical == ""
        assert not name.is_logical
        assert str(name) == "github.com/org/repo"

    def test_from_prefixed_logical(self) -> None:
        name = RepositoryName.from_prefixed("gitlab.com/org/repo/logical")
        assert name.prefix == "gitlab.com"
        assert name.owner == "org"
        assert name.physical == "repo"
        assert name.logical == "logical"
        assert name.is_logical
        assert str(name) == "gitlab.com/org/repo/logical"

    def test_with_logical(self) -> None:
        name = RepositoryName.from_prefixed("gitlab.com/org/repo").with_logical("l")
        assert str(name) == "gitlab.com/org/repo/l"

    def test_unprefixed(self) -> None:
        name = RepositoryName.from_prefixed("gitlab.com/org/repo")
        assert name.unprefixed == "org/repo"

        name = RepositoryName(None, "org2", "repo2", None)
        assert name.unprefixed == "org2/repo2"

    def test_unprefixed_logical(self) -> None:
        name = RepositoryName.from_prefixed("gitlab.com/org/repo/logic")
        assert name.unprefixed == "org/repo/logic"

        name = RepositoryName(None, "org2", "repo2", "l2")
        assert name.unprefixed == "org2/repo2/l2"


class TestRepoIdentitiesMapper:
    def test_prefixed_names_to_identities(self) -> None:
        prefixer = mk_prefixer(repo_name_to_node={"org/repo1": 1, "org/repo2": 2})
        idents = prefixer.reference_repositories(
            ["github.com/org/repo1", "github.com/org/repo2/log"],
        )
        assert idents == [
            RepositoryReference("github.com", 1, ""),
            RepositoryReference("github.com", 2, "log"),
        ]

    def test_prefixed_names_to_identities_invalid_repo(self) -> None:
        prefixer = mk_prefixer(repo_name_to_node={"org/repo1": 1})
        with pytest.raises(ValueError):
            prefixer.reference_repositories(["github.com/org/repo2"])

    def test_identities_to_prefixed_names(self) -> None:
        prefixer = mk_prefixer(repo_node_to_name={1: "org/repo1", 2: "org/repo2"})
        names = prefixer.dereference_repositories(
            [RepositoryReference("github.com", 1, ""), RepositoryReference("github.com", 1, "l")],
        )
        assert names == [
            RepositoryName.from_prefixed(r)
            for r in ("github.com/org/repo1", "github.com/org/repo1/l")
        ]

    def test_identities_to_prefixed_names_invalid_identities(self) -> None:
        prefixer = mk_prefixer(repo_node_to_prefixed_name={1: "github.com/org/repo1"})
        assert not prefixer.dereference_repositories([RepositoryReference("github.com", 2, "")])

    async def test_with_real_prefixer(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=1, full_name="athenianco/repo-A"),
                md_factory.RepositoryFactory(node_id=2, full_name="athenianco/repo-B"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)
            prefixer = await Prefixer.load((DEFAULT_MD_ACCOUNT_ID,), mdb_rw, None)

        names = prefixer.dereference_repositories(
            [
                RepositoryReference("github.com", 1, ""),
                RepositoryReference("github.com", 2, "l1"),
                RepositoryReference("github.com", 2, "l2"),
            ],
        )
        assert names == [
            RepositoryName.from_prefixed(r)
            for r in (
                "github.com/athenianco/repo-A",
                "github.com/athenianco/repo-B/l1",
                "github.com/athenianco/repo-B/l2",
            )
        ]

        identities = prefixer.reference_repositories(
            ["github.com/athenianco/repo-A/l", "github.com/athenianco/repo-B"],
        )
        assert identities == [
            RepositoryReference("github.com", 1, "l"),
            RepositoryReference("github.com", 2, ""),
        ]


def mk_prefixer(**kwargs: Any) -> Prefixer:
    """Construct a Prefixer to be used in tests."""

    if "repo_node_to_prefixed_name" in kwargs and "repo_node_to_name" not in kwargs:
        kwargs["repo_node_to_name"] = {
            k: RepositoryName.from_prefixed(v).unprefixed
            for k, v in kwargs["repo_node_to_prefixed_name"].items()
        }

    for field in dataclasses.fields(Prefixer):
        if field.name != "do_not_construct_me_directly":
            kwargs.setdefault(field.name, {})
    kwargs["do_not_construct_me_directly"] = None
    return Prefixer(**kwargs)
