import pytest

from athenian.api.internal.settings import LogicalRepositorySettings


@pytest.fixture(scope="session")
def logical_settings_labels():
    return LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {
                "labels": ["enhancement", "performance", "plumbing", "ssh", "documentation"],
            },
            "src-d/go-git/beta": {"labels": ["bug", "windows"]},
        },
        {},
    )
