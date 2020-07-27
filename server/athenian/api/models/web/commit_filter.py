from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class CommitFilter(Model):
    """Common parts of the commit filter."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "in_": List[str],
        "with_author": List[str],
        "with_committer": List[str],
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "in_": "in",
        "with_author": "with_author",
        "with_committer": "with_committer",
    }

    __slots__ = ["_" + k for k in openapi_types]

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        in_: Optional[List[str]] = None,
        with_author: Optional[List[str]] = None,
        with_committer: Optional[List[str]] = None,
    ):
        """CommitFilter - a model defined in OpenAPI

        :param account: The account of this CommitFilter.
        :param date_from: The date_from of this CommitFilter.
        :param date_to: The date_to of this CommitFilter.
        :param timezone: The timezone of this CommitFilter.
        :param in_: The in of this CommitFilter.
        :param with_author: The with_author of this CommitFilter.
        :param with_committer: The with_committer of this CommitFilter.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._in_ = in_
        self._with_author = with_author
        self._with_committer = with_committer

    @property
    def account(self) -> int:
        """Gets the account of this CommitFilter.

        Session account ID.

        :return: The account of this CommitFilter.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this CommitFilter.

        Session account ID.

        :param account: The account of this CommitFilter.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this CommitFilter.

        Commits must be made later than or equal to this date.

        :return: The date_from of this CommitFilter.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this CommitFilter.

        Commits must be made later than or equal to this date.

        :param date_from: The date_from of this CommitFilter.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this CommitFilter.

        Commits must be made earlier than or equal to this date.

        :return: The date_to of this CommitFilter.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this CommitFilter.

        Commits must be made earlier than or equal to this date.

        :param date_to: The date_to of this CommitFilter.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this CommitFilter.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this CommitFilter.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this CommitFilter.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this CommitFilter.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`")
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`")

        self._timezone = timezone

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this CommitFilter.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :return: The in_ of this CommitFilter.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this CommitFilter.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :param in_: The in_ of this CommitFilter.
        """
        if in_ is None:
            raise ValueError("Invalid value for `in`, must not be `None`")

        self._in_ = in_

    @property
    def with_author(self) -> List[str]:
        """Gets the with_author of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :return: The with_author of this CommitFilter.
        """
        return self._with_author

    @with_author.setter
    def with_author(self, with_author: List[str]):
        """Sets the with_author of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :param with_author: The with_author of this CommitFilter.
        """
        self._with_author = with_author

    @property
    def with_committer(self) -> List[str]:
        """Gets the with_committer of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :return: The with_committer of this CommitFilter.
        """
        return self._with_committer

    @with_committer.setter
    def with_committer(self, with_committer: List[str]):
        """Sets the with_committer of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :param with_committer: The with_committer of this CommitFilter.
        """
        self._with_committer = with_committer
