from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filtered_code_check_run import FilteredCodeCheckRun


class FilteredCodeCheckRuns(Model):
    """Response from `/filter/code_checks`, found code check runs ordered by repository name and \
    then by title. Note: we always consider the completed run executions only."""

    attribute_types = {
        "timeline": List[date],
        "items": List[FilteredCodeCheckRun],
    }

    attribute_map = {
        "timeline": "timeline",
        "items": "items",
    }

    def __init__(self,
                 timeline: Optional[List[date]] = None,
                 items: Optional[List[FilteredCodeCheckRun]] = None):
        """FilteredCodeCheckRuns - a model defined in OpenAPI

        :param timeline: The timeline of this FilteredCodeCheckRuns.
        :param items: The items of this FilteredCodeCheckRuns.
        """
        self._timeline = timeline
        self._items = items

    @property
    def timeline(self) -> List[date]:
        """Gets the timeline of this FilteredCodeCheckRuns.

        Sequence of dates from `date_from` till `date_to`. We choose such an interval that
        the number of items is approximately 50.

        :return: The timeline of this FilteredCodeCheckRuns.
        """
        return self._timeline

    @timeline.setter
    def timeline(self, timeline: List[date]):
        """Sets the timeline of this FilteredCodeCheckRuns.

        Sequence of dates from `date_from` till `date_to`. We choose such an interval that
        the number of items is approximately 50.

        :param timeline: The timeline of this FilteredCodeCheckRuns.
        """
        if timeline is None:
            raise ValueError("Invalid value for `timeline`, must not be `None`")

        self._timeline = timeline

    @property
    def items(self) -> List[FilteredCodeCheckRun]:
        """Gets the items of this FilteredCodeCheckRuns.

        Found check runs and their stats.

        :return: The items of this FilteredCodeCheckRuns.
        """
        return self._items

    @items.setter
    def items(self, items: List[FilteredCodeCheckRun]):
        """Sets the items of this FilteredCodeCheckRuns.

        Found check runs and their stats.

        :param items: The items of this FilteredCodeCheckRuns.
        """
        if items is None:
            raise ValueError("Invalid value for `items`, must not be `None`")

        self._items = items
