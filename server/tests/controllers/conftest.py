import pytest

from athenian.api.controllers.settings import Match, ReleaseMatchSetting


@pytest.fixture(scope="module")
def release_match_setting_tag():
    return {
        "github.com/src-d/go-git": ReleaseMatchSetting(branches="", tags=".*", match=Match.tag),
    }
