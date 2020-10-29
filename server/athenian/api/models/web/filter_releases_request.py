from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.release_with import ReleaseWith


class FilterReleasesRequest(Model, CommonFilterPropertiesMixin):
    """Structure to specify the filter traits of releases."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "in_": List[str],
        "with_": Optional[ReleaseWith],
        "jira": Optional[JIRAFilter],
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "in_": "in",
        "with_": "with",
        "jira": "jira",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        in_: Optional[List[str]] = None,
        with_: Optional[ReleaseWith] = None,
        jira: Optional[JIRAFilter] = None,
    ):
        """FilterReleasesRequest - a model defined in OpenAPI

        :param account: The account of this FilterReleasesRequest.
        :param date_from: The date_from of this FilterReleasesRequest.
        :param date_to: The date_to of this FilterReleasesRequest.
        :param timezone: The timezone of this FilterReleasesRequest.
        :param in_: The in of this FilterReleasesRequest.
        :param with_: The with of this FilterReleasesRequest.
        :param jira: The jira of this FilterReleasesRequest.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._in_ = in_
        self._with_ = with_
        self._jira = jira

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterReleasesRequest.

        :return: The in_ of this FilterReleasesRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterReleasesRequest.

        :param in_: The in_ of this FilterReleasesRequest.
        """
        self._in_ = in_

    @property
    def with_(self) -> Optional[ReleaseWith]:
        """Gets the with_ of this FilterReleasesRequest.

        Release contribution roles.

        :return: The with_ of this FilterReleasesRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[ReleaseWith]):
        """Sets the with_ of this FilterReleasesRequest.

        Release contribution roles.

        :param with_: The with_ of this FilterReleasesRequest.
        """
        self._with_ = with_

    @property
    def jira(self) -> Optional[JIRAFilter]:
        """Gets the jira of this FilterReleasesRequest.

        :return: The jira of this FilterReleasesRequest.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[JIRAFilter]):
        """Sets the jira of this FilterReleasesRequest.

        :param jira: The jira of this FilterReleasesRequest.
        """
        self._jira = jira
