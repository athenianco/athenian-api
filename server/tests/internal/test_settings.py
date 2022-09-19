import pytest

from athenian.api.internal.settings import Settings
from athenian.api.response import ResponseError


class TestSettingsListReleaseMatches:
    async def test_logical_fail(self, sdb, mdb, logical_settings_db):
        settings = Settings.from_account(1, sdb, mdb, None, None)
        with pytest.raises(ResponseError, match="424"):
            await settings.list_release_matches(["github.com/src-d/go-git/alpha"])

    async def test_logical_success(
        self,
        sdb,
        mdb,
        logical_settings_db,
        release_match_setting_tag_logical_db,
    ):
        settings = Settings.from_account(1, sdb, mdb, None, None)
        await settings.list_release_matches(["github.com/src-d/go-git/alpha"])
