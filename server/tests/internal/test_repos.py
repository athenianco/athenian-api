from athenian.api.internal.account import RepositoryReference
from athenian.api.internal.prefixer import RepositoryName
from athenian.api.internal.repos import (
    dereference_db_repositories,
    dump_db_repositories,
    parse_db_repositories,
)
from athenian.api.internal.settings import LogicalRepositorySettings
from tests.controllers.test_prefixer import mk_prefixer


class TestParseDBRepositories:
    def test_unset(self) -> None:
        assert parse_db_repositories(None) is None

    def test_empty(self) -> None:
        assert parse_db_repositories([]) == []

    def test_base(self) -> None:
        val: list[list] = [[123, None], [456, "logical"]]
        identities = parse_db_repositories(val)
        assert identities is not None
        assert all(isinstance(ident, RepositoryReference) for ident in identities)
        assert identities[0].node_id == 123
        assert identities[0].logical_name == ""
        assert identities[1].node_id == 456
        assert identities[1].logical_name == "logical"


class TestDumpDBRepositories:
    def test_none(self) -> None:
        assert dump_db_repositories(None) is None

    def test_some_identities(self) -> None:
        idents = [
            RepositoryReference("github.com", 1, "a"),
            RepositoryReference("github.com", 2, ""),
        ]
        assert dump_db_repositories(idents) == [(1, "a"), (2, "")]


class TestDereferenceDBRepositories:
    def test_empty(self, logical_settings_full) -> None:
        prefixer = mk_prefixer()
        assert dereference_db_repositories([], prefixer, logical_settings_full) == ()

    def test_base(self) -> None:
        prefixer = mk_prefixer(
            repo_node_to_prefixed_name={
                1: "github.com/athenianco/a",
                2: "github.com/athenianco/b",
            },
        )
        logical_settings = LogicalRepositorySettings(
            {
                "athenianco/a": {"labels": ["a"]},
                "athenianco/b": {"labels": ["b"]},
                "athenianco/b/logic": {"labels": ["logic"]},
            },
            {},
        )

        res = dereference_db_repositories(
            [(1, None), (2, None), (2, "logic"), (2, "xxx")], prefixer, logical_settings,
        )

        assert res == (
            RepositoryName("github.com", "athenianco", "a", ""),
            RepositoryName("github.com", "athenianco", "b", ""),
            RepositoryName("github.com", "athenianco", "b", "logic"),
        )

    def test_unknown_ids_are_ignored(self) -> None:
        prefixer = mk_prefixer(repo_node_to_prefixed_name={1: "github.com/athenianco/a"})
        logical_settings = LogicalRepositorySettings({"athenianco/a": {"labels": ["a"]}}, {})
        res = dereference_db_repositories([(1, None), (2, None)], prefixer, logical_settings)

        assert res == (RepositoryName("github.com", "athenianco", "a", ""),)
