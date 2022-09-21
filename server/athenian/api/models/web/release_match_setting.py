import re
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_match_strategy import ReleaseMatchStrategy


class _ReleaseMatchSetting(Model, sealed=False):
    branches: str
    tags: str
    events: Optional[str]
    match: str
    default_branch: Optional[str]

    def validate_branches(self, branches: str) -> str:
        """Sets the branches of this ReleaseMatchSetting.

        Regular expression to match branch names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :param branches: The branches of this ReleaseMatchSetting.
        """  # noqa
        if branches is None:
            raise ValueError("Invalid value for `branches`, must not be `None`")
        try:
            re.compile(branches)
        except re.error:
            raise ValueError("Invalid value for `branches`, must be a valid regular expression")

        return branches

    def validate_tags(self, tags: str) -> str:
        """Sets the tags of this ReleaseMatchSetting.

        Regular expression to match tag names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :param tags: The tags of this ReleaseMatchSetting.
        """  # noqa
        if tags is None:
            raise ValueError("Invalid value for `tags`, must not be `None`")
        try:
            re.compile(tags)
        except re.error:
            raise ValueError("Invalid value for `tags`, must be a valid regular expression")

        return tags

    def validate_events(self, events: str) -> str:
        """Sets the events of this ReleaseMatchSetting.

        Regular expression to match release event names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :param events: The events of this ReleaseMatchSetting.
        """  # noqa
        if events is not None:
            try:
                re.compile(events)
            except re.error:
                raise ValueError("Invalid value for `events`, must be a valid regular expression")

        return events

    def validate_match(self, match: str) -> str:
        """Sets the match of this ReleaseMatchSetting.

        :param match: The match of this ReleaseMatchSetting.
        """
        if match is None:
            raise ValueError("Invalid value for `match`, must not be `None`")
        if match not in ReleaseMatchStrategy:
            raise ValueError(
                "Invalid value for `match` (%s), must be one of %s"
                % (match, list(ReleaseMatchStrategy)),
            )

        return match


class ReleaseMatchSetting(_ReleaseMatchSetting):
    """Release matching setting for a specific repository."""

    @classmethod
    def from_dataclass(cls, struct) -> "ReleaseMatchSetting":
        """Convert a dataclass structure to the web model."""
        return ReleaseMatchSetting(
            branches=struct.branches,
            tags=struct.tags,
            events=struct.events,
            match=struct.match.name,
        )
