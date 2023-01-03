import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    Settings,
)
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import ReleaseMatchStrategy
from athenian.api.response import ResponseError
from tests.testutils.db import models_insert
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.state import (
    LogicalRepositoryFactory,
    ReleaseSettingFactory,
    RepositorySetFactory,
)


class TestSettingsListReleaseMatches:
    _TAG = ReleaseMatch[ReleaseMatchStrategy.TAG]
    _EVENT = ReleaseMatch[ReleaseMatchStrategy.EVENT]
    _BRANCH = ReleaseMatch[ReleaseMatchStrategy.BRANCH]

    async def test_logical_fail(self, sdb, mdb, logical_settings_db):
        with pytest.raises(ResponseError, match="424"):
            await self._list_release_matches(sdb, mdb, ["github.com/src-d/go-git/alpha"])

    async def test_logical_success(
        self,
        sdb,
        mdb,
        logical_settings_db,
        release_match_setting_tag_logical_db,
    ):
        rel_settings = await self._list_release_matches(
            sdb, mdb, ["github.com/src-d/go-git/alpha"],
        )
        assert sorted(rel_settings.prefixed) == [
            "github.com/src-d/go-git",
            "github.com/src-d/go-git/alpha",
        ]
        assert sorted(rel_settings.native) == ["src-d/go-git", "src-d/go-git/alpha"]

        assert rel_settings.prefixed["github.com/src-d/go-git/alpha"] == ReleaseMatchSetting(
            "master", ".*", ".*", self._TAG,
        )

    async def test_all(self, sdb: Database, mdb: Database):
        await models_insert(
            sdb,
            ReleaseSettingFactory(repo_id=40550, branches="prod", tags="^v.*", match=self._BRANCH),
            ReleaseSettingFactory(repo_id=39652769, events="ev-.*", match=self._EVENT),
            ReleaseSettingFactory(repo_id=39652699, events="ev-.*", match=self._EVENT),
        )
        rel_settings = await self._list_release_matches(sdb, mdb)

        assert sorted(rel_settings.prefixed) == [
            "github.com/src-d/gitbase",
            "github.com/src-d/go-git",
        ]
        assert sorted(rel_settings.native) == ["src-d/gitbase", "src-d/go-git"]

        assert rel_settings.prefixed["github.com/src-d/gitbase"] == ReleaseMatchSetting(
            "{{default}}", ".*", "ev-.*", self._EVENT,
        )
        assert rel_settings.prefixed["github.com/src-d/go-git"] == ReleaseMatchSetting(
            "prod", "^v.*", ".*", self._BRANCH,
        )

    async def test_logical_repos(self, sdb: Database, mdb: Database) -> None:
        await sdb.execute(sa.delete(RepositorySet))
        await models_insert(
            sdb,
            LogicalRepositoryFactory(name="alpha", repository_id=40550),
            LogicalRepositoryFactory(name="beta", repository_id=40550),
            RepositorySetFactory(
                items=[
                    ["github.com", 40550, ""],
                    ["github.com", 40550, "alpha"],
                    ["github.com", 40550, "beta"],
                    ["github.com", 39652769, ""],
                ],
            ),
            ReleaseSettingFactory(repo_id=40550, branches="prod", tags="^v.*", match=self._BRANCH),
            ReleaseSettingFactory(repo_id=39652769, events="ev-.*", match=self._EVENT),
            ReleaseSettingFactory(
                repo_id=40550,
                logical_name="alpha",
                tags="^valpha.*",
                match=self._TAG,
            ),
            ReleaseSettingFactory(
                repo_id=40550,
                logical_name="beta",
                branches="prod",
                match=self._BRANCH,
            ),
        )
        rel_settings = await self._list_release_matches(sdb, mdb)
        assert sorted(rel_settings.prefixed) == [
            "github.com/src-d/gitbase",
            "github.com/src-d/go-git",
            "github.com/src-d/go-git/alpha",
            "github.com/src-d/go-git/beta",
        ]
        assert sorted(rel_settings.native) == [
            "src-d/gitbase",
            "src-d/go-git",
            "src-d/go-git/alpha",
            "src-d/go-git/beta",
        ]
        assert rel_settings.prefixed["github.com/src-d/go-git/alpha"] == ReleaseMatchSetting(
            "{{default}}", "^valpha.*", ".*", self._TAG,
        )
        assert rel_settings.prefixed["github.com/src-d/go-git/beta"] == ReleaseMatchSetting(
            "prod", ".*", ".*", self._BRANCH,
        )

        rel_settings = await self._list_release_matches(sdb, mdb, ["github.com/src-d/go-git/beta"])
        # physical repo settings are included anyway, sibling logical repo settings not
        assert sorted(rel_settings.prefixed) == [
            "github.com/src-d/go-git",
            "github.com/src-d/go-git/beta",
        ]
        assert rel_settings.prefixed["github.com/src-d/go-git/beta"] == ReleaseMatchSetting(
            "prod", ".*", ".*", self._BRANCH,
        )
        assert rel_settings.prefixed["github.com/src-d/go-git"] == ReleaseMatchSetting(
            "prod", "^v.*", ".*", self._BRANCH,
        )

    async def _list_release_matches(self, sdb, mdb, repos=None) -> ReleaseSettings:
        prefixer = await Prefixer.load((DEFAULT_MD_ACCOUNT_ID,), mdb, None)
        settings = Settings.from_account(1, prefixer, sdb, mdb, None, None)
        return await settings.list_release_matches(repos)


class TestLogicalRepositorySettings:
    def test_augment_with_logical_repos(self) -> None:
        settings = LogicalRepositorySettings(
            {
                "src-d/go-git/alpha": {"title": r"^alpha.*"},
                "src-d/go-git/beta": {"title": r"^beta.*"},
            },
            {},
        )
        exploded = settings.augment_with_logical_repos(["src-d/go-git", "src-d/foobar"])
        assert exploded[0] == "src-d/go-git"
        assert sorted(exploded[1:3]) == ["src-d/go-git/alpha", "src-d/go-git/beta"]
        assert exploded[3] == "src-d/foobar"

        exploded = settings.augment_with_logical_repos(["src-d/go-git"])
        assert exploded[0] == "src-d/go-git"
        assert sorted(exploded[1:3]) == ["src-d/go-git/alpha", "src-d/go-git/beta"]

        exploded = settings.augment_with_logical_repos(["src-d/go-git", "src-d/go-git/alpha"])
        assert exploded == ["src-d/go-git", "src-d/go-git/alpha"]

        exploded = settings.augment_with_logical_repos(["src-d/go-git/alpha"])
        assert exploded == ["src-d/go-git/alpha"]
