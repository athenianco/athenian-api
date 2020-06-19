from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class FilterContributorsRequest(Model):
    """Filters for `/filter/contributors`."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "in_": List[str],
        "as_": List[str],
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "in_": "in",
        "as_": "as",
    }

    def __init__(
            self,
            account: Optional[int] = None,
            date_from: Optional[date] = None,
            date_to: Optional[date] = None,
            timezone: Optional[int] = None,
            in_: Optional[List[str]] = None,
            as_: List[str] = None,
    ):
        """FilterContributorsRequest - a model defined in OpenAPI

        :param account: The account of this FilterContributorsRequest.
        :param date_from: The date_from of this FilterContributorsRequest.
        :param date_to: The date_to of this FilterContributorsRequest.
        :param timezone: The timezone of this FilterContributorsRequest.
        :param in_: The in_ of this FilterContributorsRequest.
        :param as_: The as_ of this FilterContributorsRequest.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._in_ = in_
        self._as_ = as_

    @property
    def account(self) -> int:
        """Gets the account of this FilterContributorsRequest.

        Session account ID.

        :return: The account of this FilterContributorsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this FilterContributorsRequest.

        Session account ID.

        :param account: The account of this FilterContributorsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this FilterContributorsRequest.

        Updates must be later than or equal to this date. An update is any action that influences
        the stage assignment.

        :return: The date_from of this FilterContributorsRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this FilterContributorsRequest.

        Updates must be later than or equal to this date. An update is any action that influences
        the stage assignment.

        :param date_from: The date_from of this FilterContributorsRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this FilterContributorsRequest.

        Updates must be earlier than or equal to this date. An update is any action that influences
        the stage assignment.

        :return: The date_to of this FilterContributorsRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this FilterContributorsRequest.

        Updates must be earlier than or equal to this date. An update is any action that influences
        the stage assignment.

        :param date_to: The date_to of this FilterContributorsRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self):
        """Gets the timezone of this FilterContributorsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this FilterContributorsRequest.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this FilterContributorsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this FilterContributorsRequest.
        """
        self._timezone = timezone

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterContributorsRequest.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :return: The in_ of this FilterContributorsRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterContributorsRequest.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :param in_: The in_ of this FilterContributorsRequest.
        """
        self._in_ = in_

    @property
    def as_(self) -> List[str]:
        """Gets the as_ of this FilterContributorsRequest.

        :return: The as_ of this FilterContributorsRequest.
        """
        return self._as_

    @as_.setter
    def as_(self, as_: List[str]):
        """Sets the as_ of this FilterContributorsRequest.

        :param as_: The as_ of this FilterContributorsRequest.
        """
        allowed_values = {
            "author",
            "reviewer",
            "commit_author",
            "commit_committer",
            "commenter",
            "merger",
            "releaser",
        }
        if not set(as_).issubset(allowed_values):
            raise ValueError(
                "Invalid values for `as_` [%s], must be a subset of [%s]" % (
                    ", ".join(set(as_) - allowed_values),
                    ", ".join(allowed_values),
                ),
            )

        self._as_ = as_
