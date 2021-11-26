from typing import List, Optional

from athenian.api.models.web.account import _Account
from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.release_match_setting import _ReleaseMatchSetting


class _ReleaseMatchRequest(Model):
    openapi_types = {"repositories": List[str]}
    attribute_map = {"repositories": "repositories"}

    def __init__(self, repositories: Optional[List[str]] = None):
        """ReleaseMatchRequest - a model defined in OpenAPI

        :param repositories: The repositories of this ReleaseMatchRequest.
        """
        self._repositories = repositories

    @property
    def repositories(self) -> List[str]:
        """Gets the repositories of this ReleaseMatchSetting.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :return: The repositories of this ReleaseMatchSetting.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: List[str]):
        """Sets the repositories of this ReleaseMatchSetting.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :param repositories: The repositories of this ReleaseMatchSetting.
        """
        if repositories is None:
            raise ValueError("Invalid value for `repositories`, must not be `None`")

        self._repositories = repositories


ReleaseMatchRequest = AllOf(_ReleaseMatchSetting,
                            _ReleaseMatchRequest,
                            _Account,
                            name="ReleaseMatchRequest",
                            module=__name__)
