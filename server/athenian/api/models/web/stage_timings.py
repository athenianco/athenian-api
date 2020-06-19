from datetime import timedelta
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class StageTimings(Model):
    """Time spent by the PR in each pipeline stage."""

    openapi_types = {
        "wip": timedelta,
        "review": timedelta,
        "merge": timedelta,
        "release": timedelta,
    }

    attribute_map = {
        "wip": "wip",
        "review": "review",
        "merge": "merge",
        "release": "release",
    }

    def __init__(
        self,
        wip: Optional[timedelta] = None,
        review: Optional[timedelta] = None,
        merge: Optional[timedelta] = None,
        release: Optional[timedelta] = None,
    ):
        """StageTimings - a model defined in OpenAPI

        :param wip: The wip of this StageTimings.
        :param review: The review of this StageTimings.
        :param merge: The merge of this StageTimings.
        :param release: The release of this StageTimings.
        """
        self._wip = wip
        self._review = review
        self._merge = merge
        self._release = release

    @property
    def wip(self) -> timedelta:
        """Gets the wip of this StageTimings.

        :return: The wip of this StageTimings.
        """
        return self._wip

    @wip.setter
    def wip(self, wip: timedelta) -> None:
        """Sets the wip of this StageTimings.

        :param wip: The wip of this StageTimings.
        """
        if wip is None:
            raise ValueError("Invalid value for `wip`, must not be `None`")

        self._wip = wip

    @property
    def review(self) -> timedelta:
        """Gets the review of this StageTimings.

        :return: The review of this StageTimings.
        """
        return self._review

    @review.setter
    def review(self, review: timedelta) -> None:
        """Sets the review of this StageTimings.

        :param review: The review of this StageTimings.
        """
        self._review = review

    @property
    def merge(self) -> timedelta:
        """Gets the merge of this StageTimings.

        :return: The merge of this StageTimings.
        """
        return self._merge

    @merge.setter
    def merge(self, merge: timedelta) -> None:
        """Sets the merge of this StageTimings.

        :param merge: The merge of this StageTimings.
        """
        self._merge = merge

    @property
    def release(self) -> timedelta:
        """Gets the release of this StageTimings.

        :return: The release of this StageTimings.
        """
        return self._release

    @release.setter
    def release(self, release: timedelta) -> None:
        """Sets the release of this StageTimings.

        :param release: The release of this StageTimings.
        """
        self._release = release
