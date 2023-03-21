from datetime import datetime, timezone

import medvedi as md
from medvedi.testing import assert_frame_equal
import pytest

from athenian.api.defer import with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.commit import _empty_dag
from athenian.api.internal.miners.github.logical import (
    split_logical_deployed_components,
    split_logical_prs,
)
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)


@pytest.fixture(scope="function")
@with_defer
async def sample_prs(mdb, pdb, branches):
    prs, _, labels = await PullRequestMiner.fetch_prs(
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        True,
        None,
        None,
        branches,
        {"src-d/go-git": (True, _empty_dag())},
        1,
        (6366825,),
        mdb,
        pdb,
        None,
        with_labels=True,
    )
    return prs, labels


@pytest.fixture(scope="function")
def sample_deployments():
    deployments = md.DataFrame(
        {
            DeploymentNotification.name.name: ["one", "two", "three", "four"],
            DeploymentNotification.environment.name: ["production"] * 4,
        },
    )
    deployments.set_index(DeploymentNotification.name.name, inplace=True)
    components = md.DataFrame(
        {
            DeployedComponent.deployment_name.name: ["one", "one", "two", "two", "three", "four"],
            DeployedComponent.repository_full_name: [
                "src-d/go-git",
                "src-d/hercules",
                "src-d/go-git",
                "src-d/hercules",
                "src-d/go-git",
                "src-d/hercules",
            ],
        },
    )
    components.set_index(DeployedComponent.deployment_name.name, inplace=True)
    labels = md.DataFrame(
        {
            DeployedLabel.deployment_name.name: [
                "one",
                "one",
                "two",
                "two",
                "three",
                "three",
                "four",
            ],
            DeployedLabel.key.name: [
                "first",
                "second",
                "first",
                "third",
                "first",
                "second",
                "first",
            ],
            DeployedLabel.value.name: [
                "alpha",
                "alpha",
                "alpha",
                "beta",
                "gamma",
                "alpha",
                "gamma",
            ],
        },
    )
    labels.set_index(DeployedComponent.deployment_name.name, inplace=True)
    return deployments, components, labels


async def test_split_logical_prs_title(sample_prs):
    settings = LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"title": ".*[Ff]ix"},
            "src-d/go-git/beta": {"title": ".*[Aa]dd"},
        },
        {},
    )
    result = split_logical_prs(*sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 326
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 188
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 138
    for index, title in result.iterrows(
        PullRequest.repository_full_name.name, PullRequest.title.name,
    ):
        if index == "src-d/go-git/alpha":
            assert "fix" in title or "Fix" in title
        elif index == "src-d/go-git/beta":
            assert "add" in title or "Add" in title
        else:
            assert "fix" not in title and "Fix" not in title
            assert "add" not in title and "Add" not in title


async def test_split_logical_prs_labels(sample_prs):
    settings = LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"labels": ["bug"]},
            "src-d/go-git/beta": {"labels": ["enhancement"]},
        },
        {},
    )
    result = split_logical_prs(*sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 12
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 5
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 7


async def test_split_logical_prs_both(sample_prs):
    settings = LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"title": ".*[Ff]ix", "labels": ["bug"]},
            "src-d/go-git/beta": {"title": ".*[Aa]dd", "labels": ["enhancement"]},
        },
        {},
    )
    result = split_logical_prs(*sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 334
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 190
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 144


async def test_split_logical_prs_none(sample_prs):
    settings = LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {},
            "src-d/go-git/beta": {},
        },
        {},
    )
    result = split_logical_prs(*sample_prs, {"src-d/go-git/alpha", "src-d/go-git/beta"}, settings)
    assert len(result) == 0


async def test_split_logical_prs_mixture(sample_prs):
    settings = LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"labels": ["bug"]},
            "src-d/go-git/beta": {"labels": ["enhancement"]},
        },
        {},
    )
    result = split_logical_prs(
        *sample_prs, {"src-d/go-git", "src-d/go-git/alpha", "src-d/go-git/beta"}, settings,
    )
    assert len(result) == 676
    assert (result.index.get_level_values(1) == "src-d/go-git/alpha").sum() == 5
    assert (result.index.get_level_values(1) == "src-d/go-git/beta").sum() == 7


async def test_split_logical_deployed_components_title_one(sample_deployments):
    notifications, components, labels = sample_deployments
    logical_settings = LogicalRepositorySettings(
        {},
        {
            "src-d/go-git/alpha": {"title": "one|two"},
            "src-d/go-git/beta": {"title": "two|three"},
        },
    )
    split_components = split_logical_deployed_components(
        notifications, labels, components, ["src-d/go-git/alpha"], logical_settings,
    )
    split_components.reset_index(inplace=True)
    assert_frame_equal(
        split_components,
        md.DataFrame(
            {
                DeployedComponent.deployment_name.name: [
                    "one",
                    "two",
                    "one",
                    "two",
                    "three",
                    "four",
                ],
                DeployedComponent.repository_full_name: [
                    "src-d/go-git/alpha",
                    "src-d/go-git/alpha",
                    "src-d/hercules",
                    "src-d/hercules",
                    "src-d/go-git",
                    "src-d/hercules",
                ],
            },
        ),
    )


async def test_split_logical_deployed_components_title_two(sample_deployments):
    notifications, components, labels = sample_deployments
    logical_settings = LogicalRepositorySettings(
        {},
        {
            "src-d/go-git/alpha": {"title": "one|two"},
            "src-d/go-git/beta": {"title": "two|three"},
        },
    )
    split_components = split_logical_deployed_components(
        notifications,
        labels,
        components,
        ["src-d/go-git/alpha", "src-d/go-git/beta"],
        logical_settings,
    )
    split_components.reset_index(inplace=True)
    assert_frame_equal(
        split_components,
        md.DataFrame(
            {
                DeployedComponent.deployment_name.name: [
                    "one",
                    "two",
                    "two",
                    "three",
                    "one",
                    "two",
                    "four",
                ],
                DeployedComponent.repository_full_name: [
                    "src-d/go-git/alpha",
                    "src-d/go-git/alpha",
                    "src-d/go-git/beta",
                    "src-d/go-git/beta",
                    "src-d/hercules",
                    "src-d/hercules",
                    "src-d/hercules",
                ],
            },
        ),
    )


async def test_split_logical_deployed_components_labels_one(sample_deployments):
    notifications, components, labels = sample_deployments
    logical_settings = LogicalRepositorySettings(
        {},
        {
            "src-d/go-git/alpha": {"labels": {"first": ["alpha", "gamma"]}},
            "src-d/go-git/beta": {"labels": {"second": ["alpha"]}},
        },
    )
    split_components = split_logical_deployed_components(
        notifications, labels, components, ["src-d/go-git/alpha"], logical_settings,
    )
    split_components.reset_index(inplace=True)
    assert_frame_equal(
        split_components,
        md.DataFrame(
            {
                DeployedComponent.deployment_name.name: [
                    "one",
                    "two",
                    "three",
                    "one",
                    "two",
                    "four",
                ],
                DeployedComponent.repository_full_name: [
                    "src-d/go-git/alpha",
                    "src-d/go-git/alpha",
                    "src-d/go-git/alpha",
                    "src-d/hercules",
                    "src-d/hercules",
                    "src-d/hercules",
                ],
            },
        ),
    )


async def test_split_logical_deployed_components_labels_two(sample_deployments):
    notifications, components, labels = sample_deployments
    logical_settings = LogicalRepositorySettings(
        {},
        {
            "src-d/go-git/alpha": {"labels": {"first": ["alpha", "gamma"]}},
            "src-d/go-git/beta": {"labels": {"second": ["alpha"]}},
        },
    )
    split_components = split_logical_deployed_components(
        notifications,
        labels,
        components,
        ["src-d/go-git/alpha", "src-d/go-git/beta"],
        logical_settings,
    )
    split_components.reset_index(inplace=True)
    assert_frame_equal(
        split_components,
        md.DataFrame(
            {
                DeployedComponent.deployment_name.name: [
                    "one",
                    "two",
                    "three",
                    "one",
                    "three",
                    "one",
                    "two",
                    "four",
                ],
                DeployedComponent.repository_full_name: [
                    "src-d/go-git/alpha",
                    "src-d/go-git/alpha",
                    "src-d/go-git/alpha",
                    "src-d/go-git/beta",
                    "src-d/go-git/beta",
                    "src-d/hercules",
                    "src-d/hercules",
                    "src-d/hercules",
                ],
            },
        ),
    )


async def test_split_logical_deployed_components_labels_mixture(sample_deployments):
    notifications, components, labels = sample_deployments
    logical_settings = LogicalRepositorySettings(
        {},
        {
            "src-d/go-git/alpha": {"title": "one", "labels": {"first": ["gamma"]}},
            "src-d/go-git/beta": {"labels": {"second": ["alpha"]}},
        },
    )
    split_components = split_logical_deployed_components(
        notifications, labels, components, ["src-d/go-git/alpha"], logical_settings,
    )
    split_components.reset_index(inplace=True)
    assert_frame_equal(
        split_components,
        md.DataFrame(
            {
                DeployedComponent.deployment_name.name: [
                    "one",
                    "three",
                    "one",
                    "two",
                    "two",
                    "four",
                ],
                DeployedComponent.repository_full_name: [
                    "src-d/go-git/alpha",
                    "src-d/go-git/alpha",
                    "src-d/hercules",
                    "src-d/go-git",
                    "src-d/hercules",
                    "src-d/hercules",
                ],
            },
        ),
    )
