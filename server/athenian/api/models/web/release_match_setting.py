from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_match_strategy import ReleaseMatchStrategy


class _ReleaseMatchSetting(Model):
    __enable_slots__ = False

    openapi_types = {
        "branches": str,
        "tags": str,
        "match": str,
        "default_branch": Optional[str],
    }

    attribute_map = {
        "branches": "branches",
        "tags": "tags",
        "match": "match",
        "default_branch": "default_branch",
    }

    def __init__(self,
                 branches: Optional[str] = None,
                 tags: Optional[str] = None,
                 match: Optional[str] = None,
                 default_branch: Optional[str] = None):
        """ReleaseMatchSetting - a model defined in OpenAPI

        :param branches: The branches of this ReleaseMatchSetting.
        :param tags: The tags of this ReleaseMatchSetting.
        :param match: The match of this ReleaseMatchSetting.
        :param default_branch: The default_branch of this ReleaseMatchSetting.
        """
        self._branches = branches
        self._tags = tags
        self._match = match
        self._default_branch = default_branch

    @property
    def branches(self) -> str:
        """Gets the branches of this ReleaseMatchSetting.

        Regular expression to match branch names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :return: The branches of this ReleaseMatchSetting.
        """  # noqa
        return self._branches

    @branches.setter
    def branches(self, branches: str):
        """Sets the branches of this ReleaseMatchSetting.

        Regular expression to match branch names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :param branches: The branches of this ReleaseMatchSetting.
        """  # noqa
        if branches is None:
            raise ValueError("Invalid value for `branches`, must not be `None`")

        self._branches = branches

    @property
    def tags(self) -> str:
        """Gets the tags of this ReleaseMatchSetting.

        Regular expression to match tag names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :return: The tags of this ReleaseMatchSetting.
        """  # noqa
        return self._tags

    @tags.setter
    def tags(self, tags: str):
        """Sets the tags of this ReleaseMatchSetting.

        Regular expression to match tag names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :param tags: The tags of this ReleaseMatchSetting.
        """  # noqa
        if tags is None:
            raise ValueError("Invalid value for `tags`, must not be `None`")

        self._tags = tags

    @property
    def match(self) -> str:
        """Gets the match of this ReleaseMatchSetting.

        :return: The match of this ReleaseMatchSetting.
        """
        return self._match

    @match.setter
    def match(self, match: str):
        """Sets the match of this ReleaseMatchSetting.

        :param match: The match of this ReleaseMatchSetting.
        """
        if match is None:
            raise ValueError("Invalid value for `match`, must not be `None`")
        if match not in ReleaseMatchStrategy:
            raise ValueError(
                "Invalid value for `match` (%s), must be one of %s" %
                (match, list(ReleaseMatchStrategy)))

        self._match = match

    @property
    def default_branch(self) -> Optional[str]:
        """Gets the default_branch of this ReleaseMatchSetting.

        Name of the default branch of this repository.

        :return: The default_branch of this ReleaseMatchSetting.
        """  # noqa
        return self._default_branch

    @default_branch.setter
    def default_branch(self, default_branch: Optional[str]):
        """Sets the default_branch of this ReleaseMatchSetting.

        Name of the default branch of this repository.

        :param default_branch: The default_branch of this ReleaseMatchSetting.
        """  # noqa
        self._default_branch = default_branch


class ReleaseMatchSetting(_ReleaseMatchSetting):
    """Release matching setting for a specific repository."""

    __enable_slots__ = True

    @classmethod
    def from_dataclass(cls, struct) -> "ReleaseMatchSetting":
        """Convert a dataclass structure to the web model."""
        return ReleaseMatchSetting(branches=struct.branches,
                                   tags=struct.tags,
                                   match=struct.match.name)
