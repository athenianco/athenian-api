from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_property import PullRequestProperty
from athenian.api.models.web.pull_request_with import PullRequestWith


class FilterPullRequestsRequest(Model):
    """PR filters for /filter/pull_requests."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "in_": List[str],
        "properties": List[str],
        "with_": PullRequestWith,
        "labels": List[str],
        "exclude_inactive": bool,
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "in_": "in",
        "properties": "properties",
        "with_": "with",
        "labels": "labels",
        "exclude_inactive": "exclude_inactive",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        in_: Optional[List[str]] = None,
        properties: Optional[List[str]] = None,
        with_: Optional[PullRequestWith] = None,
        labels: Optional[List[str]] = None,
        exclude_inactive: Optional[bool] = None,
    ):
        """FilterPullRequestsRequest - a model defined in OpenAPI

        :param account: The account of this FilterPullRequestsRequest.
        :param date_from: The date_from of this FilterPullRequestsRequest.
        :param date_to: The date_to of this FilterPullRequestsRequest.
        :param timezone: The timezone of this FilterPullRequestsRequest.
        :param in_: The in_ of this FilterPullRequestsRequest.
        :param properties: The properties of this FilterPullRequestsRequest.
        :param with_: The with_ of this FilterPullRequestsRequest.
        :param labels: The labels of this FilterPullRequestsRequest.
        :param exclude_inactive: The exclude_inactive of this FilterPullRequestsRequest.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._in_ = in_
        self._properties = properties
        self._with_ = with_
        self._labels = labels
        self._exclude_inactive = exclude_inactive

    @property
    def account(self) -> int:
        """Gets the account of this FilterPullRequestsRequest.

        Session account ID.

        :return: The account of this FilterPullRequestsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this FilterPullRequestsRequest.

        Session account ID.

        :param account: The account of this FilterPullRequestsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this FilterPullRequestsRequest.

        PRs must be updated later than or equal to this date.

        :return: The date_from of this FilterPullRequestsRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this FilterPullRequestsRequest.

        PRs must be updated later than or equal to this date.

        :param date_from: The date_from of this FilterPullRequestsRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this FilterPullRequestsRequest.

        PRs must be updated earlier than or equal to this date.

        :return: The date_to of this FilterPullRequestsRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this FilterPullRequestsRequest.

        PRs must be updated earlier than or equal to this date.

        :param date_to: The date_to of this FilterPullRequestsRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this FilterPullRequestsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this FilterPullRequestsRequest.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this FilterPullRequestsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this FilterPullRequestsRequest.
        """
        self._timezone = timezone

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterPullRequestsRequest.

        :return: The in_ of this FilterPullRequestsRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in of this FilterPullRequestsRequest.

        :param in_: The in of this FilterPullRequestsRequest.
        """
        if in_ is None:
            raise ValueError("Invalid value for `in`, must not be `None`")

        self._in_ = in_

    @property
    def properties(self) -> List[str]:
        """Gets the properties of this FilterPullRequestsRequest.

        :return: The properties of this FilterPullRequestsRequest.
        """
        return self._properties

    @properties.setter
    def properties(self, properties: List[str]):
        """Sets the properties of this FilterPullRequestsRequest.

        :param properties: The properties of this FilterPullRequestsRequest.
        """
        if properties is None:
            raise ValueError("Invalid value for `properties`, must not be `None`")

        for stage in properties:
            if stage not in PullRequestProperty:
                raise ValueError("Invalid property: %s" % stage)

        self._properties = properties

    @property
    def with_(self) -> Optional[PullRequestWith]:
        """Gets the with_ of this FilterPullRequestsRequest.

        :return: The with_ of this FilterPullRequestsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: PullRequestWith):
        """Sets the with_ of this FilterPullRequestsRequest.

        :param with_: The with_ of this FilterPullRequestsRequest.
        """
        self._with_ = with_

    @property
    def labels(self) -> Optional[List[str]]:
        """Gets the labels of this FilterPullRequestsRequest.

        :return: The labels of this FilterPullRequestsRequest.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: List[str]):
        """Sets the labels of this FilterPullRequestsRequest.

        :param labels: The labels of this FilterPullRequestsRequest.
        """
        self._labels = labels

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this FilterPullRequestsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :return: The exclude_inactive of this FilterPullRequestsRequest.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this FilterPullRequestsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :param exclude_inactive: The exclude_inactive of this FilterPullRequestsRequest.
        """
        self._exclude_inactive = exclude_inactive
