from datetime import date
from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.filter_jira_common import FilterJIRACommon
from athenian.api.models.web.jira_filter_return import JIRAFilterReturn
from athenian.api.models.web.jira_filter_with import JIRAFilterWith


class FilterJIRAStuffSpecials(Model, sealed=False):
    """Request of `/filter/jira` to retrieve epics and labels."""

    with_: (Optional[JIRAFilterWith], "with")
    return_: (Optional[list[str]], "return")

    # We have to redefine `date_from` and `date_to` to assign Optional and allow null-s.

    def validate_date_from(self, date_from: Optional[date]) -> Optional[date]:
        """Sets the date_from of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :param date_from: The date_from of this Model.
        """
        return date_from

    def validate_date_to(self, date_to: Optional[date]) -> Optional[date]:
        """Sets the date_to of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :param date_to: The date_to of this Model.
        """
        return date_to

    def validate_return_(self, return_: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the return of this FilterJIRAStuff.

        Specifies which items are required, an empty or missing array means everything.

        :param return_: The return of this FilterJIRAStuff.
        """
        if diff := set(return_ or []) - set(JIRAFilterReturn):
            raise ValueError("`return` contains invalid values: %s" % diff)
        return return_


FilterJIRAStuff = AllOf(
    FilterJIRAStuffSpecials, FilterJIRACommon, name="FilterJIRAStuff", module=__name__,
)
