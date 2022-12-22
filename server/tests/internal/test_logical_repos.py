from athenian.api.internal.logical_repos import (
    coerce_logical_repos,
    contains_logical_repos,
    drop_logical_repo,
)


def test_drop_logical_repo() -> None:
    assert drop_logical_repo("src-d/go-git") == "src-d/go-git"
    assert drop_logical_repo("src-d/go-git/alpha") == "src-d/go-git"
    assert drop_logical_repo("") == ""


def test_coerce_logical_repos() -> None:
    assert coerce_logical_repos(["src-d/go-git"]) == {"src-d/go-git": {"src-d/go-git"}}
    assert coerce_logical_repos(["src-d/go-git/alpha", "src-d/go-git/beta"]) == {
        "src-d/go-git": {"src-d/go-git/alpha", "src-d/go-git/beta"},
    }
    assert coerce_logical_repos([]) == {}


def test_contains_logical_repos() -> None:
    assert not contains_logical_repos([])
    assert not contains_logical_repos(["src-d/go-git"])
    assert contains_logical_repos(["src-d/go-git/"])
    assert contains_logical_repos(["src-d/go-git/alpha"])
