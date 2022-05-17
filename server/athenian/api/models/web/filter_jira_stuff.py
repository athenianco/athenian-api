from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.filter_jira_common import FilterJIRACommon
from athenian.api.models.web.jira_filter_return import JIRAFilterReturn
from athenian.api.models.web.jira_filter_with import JIRAFilterWith


class FilterJIRAStuffSpecials(Model, sealed=False):
    """Request of `/filter/jira` to retrieve epics and labels."""

    attribute_types = {
        "with_": Optional[JIRAFilterWith],
        "return_": Optional[List[str]],
    }

    attribute_map = {
        "with_": "with",
        "return_": "return",
    }

    def __init__(self,
                 with_: Optional[JIRAFilterWith] = None,
                 return_: Optional[List[str]] = None,
                 ):
        """FilterJIRAStuff - a model defined in OpenAPI

        :param with_: The with_ of this FilterJIRAStuff.
        :param return_: The return of this FilterJIRAStuff.
        """
        self._with_ = with_
        self._return_ = return_

    # We have to redefine `date_from` and `date_to` to assign Optional and allow null-s.

    @property
    def date_from(self) -> Optional[date]:
        """Gets the date_from of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :return: The date_from of this Model.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: Optional[date]) -> None:
        """Sets the date_from of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :param date_from: The date_from of this Model.
        """
        self._date_from = date_from

    @property
    def date_to(self) -> Optional[date]:
        """Gets the date_to of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :return: The date_to of this Model.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: Optional[date]) -> None:
        """Sets the date_to of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :param date_to: The date_to of this Model.
        """
        self._date_to = date_to

    @property
    def return_(self) -> Optional[List[str]]:
        """Gets the return of this FilterJIRAStuff.

        Specifies which items are required, an empty or missing array means everything.

        :return: The return of this FilterJIRAStuff.
        """
        return self._return_

    @return_.setter
    def return_(self, return_: Optional[List[str]]) -> None:
        """Sets the return of this FilterJIRAStuff.

        Specifies which items are required, an empty or missing array means everything.

        :param return_: The return of this FilterJIRAStuff.
        """
        if diff := set(return_ or []) - set(JIRAFilterReturn):
            raise ValueError("`return` contains invalid values: %s" % diff)
        self._return_ = return_

    @property
    def with_(self) -> Optional[JIRAFilterWith]:
        """Gets the with of this FilterJIRAStuff.

        JIRA issue participants.

        :return: The with of this FilterJIRAStuff.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[JIRAFilterWith]):
        """Sets the with of this FilterJIRAStuff.

        JIRA issue participants.

        :param with_: The with of this FilterJIRAStuff.
        """
        self._with_ = with_


FilterJIRAStuff = AllOf(FilterJIRAStuffSpecials, FilterJIRACommon,
                        name="FilterJIRAStuff", module=__name__)
