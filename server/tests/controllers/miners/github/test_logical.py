from datetime import datetime, timezone

import pytest

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.commit import _empty_dag
from athenian.api.controllers.miners.github.logical import split_logical_repositories
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.settings import LogicalRepositorySettings
from athenian.api.defer import with_defer


@pytest.fixture(scope="function")
@with_defer
async def sample_prs(mdb, pdb, branches):
    prs, _, labels = await PullRequestMiner.fetch_prs(
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        True, None, None, branches, {"src-d/go-git": _empty_dag()}, 1, (6366825,),
        mdb, pdb, None, with_labels=True)
    return prs, labels


async def test_split_logical_repositories_title(sample_prs):
    settings = LogicalRepositorySettings({
        "src-d/go-git/alpha": {"title": ".*[Ff]ix"},
        "src-d/go-git/beta": {"title": ".*[Aa]dd"},
    }, {}, {})
    result = split_logical_repositories(
        *sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 326
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 188
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 138
    for row in result.itertuples():
        if row.Index[1] == "src-d/go-git/alpha":
            assert "fix" in row.title or "Fix" in row.title
        elif row.Index[1] == "src-d/go-git/beta":
            assert "add" in row.title or "Add" in row.title
        else:
            assert "fix" not in row.title and "Fix" not in row.title
            assert "add" not in row.title and "Add" not in row.title


async def test_split_logical_repositories_labels(sample_prs):
    settings = LogicalRepositorySettings({
        "src-d/go-git/alpha": {"labels": ["bug"]},
        "src-d/go-git/beta": {"labels": ["enhancement"]},
    }, {}, {})
    result = split_logical_repositories(
        *sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 12
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 5
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 7


async def test_split_logical_repositories_both(sample_prs):
    settings = LogicalRepositorySettings({
        "src-d/go-git/alpha": {"title": ".*[Ff]ix", "labels": ["bug"]},
        "src-d/go-git/beta": {"title": ".*[Aa]dd", "labels": ["enhancement"]},
    }, {}, {})
    result = split_logical_repositories(
        *sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 334
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 190
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 144


async def test_split_logical_repositories_none(sample_prs):
    settings = LogicalRepositorySettings({
        "src-d/go-git/alpha": {},
        "src-d/go-git/beta": {},
    }, {}, {})
    result = split_logical_repositories(
        *sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 0


async def test_split_logical_repositories_mixture(sample_prs):
    settings = LogicalRepositorySettings({
        "src-d/go-git/alpha": {"labels": ["bug"]},
        "src-d/go-git/beta": {"labels": ["enhancement"]},
    }, {}, {})
    result = split_logical_repositories(
        *sample_prs, {"src-d/go-git", "src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 676
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 5
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 7
