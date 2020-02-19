from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class FilterItemsRequest(Model):
    """Structure to specify the filtering traits for repositories and contributors."""

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        in_: Optional[List[str]] = None,
    ):
        """FilterItemsRequest - a model defined in OpenAPI

        :param account: The account of this FilterItemsRequest.
        :param date_from: The date_from of this FilterItemsRequest.
        :param date_to: The date_to of this FilterItemsRequest.
        :param in_: The in of this FilterItemsRequest.
        """
        self.openapi_types = {
            "account": int,
            "date_from": date,
            "date_to": date,
            "in_": List[str],
        }

        self.attribute_map = {
            "account": "account",
            "date_from": "date_from",
            "date_to": "date_to",
            "in_": "in",
        }

        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._in_ = in_

    @property
    def account(self) -> int:
        """Gets the account of this FilterItemsRequest.

        Session account ID.

        :return: The account of this FilterItemsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this FilterItemsRequest.

        Session account ID.

        :param account: The account of this FilterItemsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this FilterItemsRequest.

        Updates must be later than or equal to this date.

        :return: The date_from of this FilterItemsRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this FilterItemsRequest.

        Updates must be later than or equal to this date.

        :param date_from: The date_from of this FilterItemsRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this FilterItemsRequest.

        Updates must be earlier than or equal to this date.

        :return: The date_to of this FilterItemsRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this FilterItemsRequest.

        Updates must be earlier than or equal to this date.

        :param date_to: The date_to of this FilterItemsRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterItemsRequest.

        :return: The in_ of this FilterItemsRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterItemsRequest.

        :param in_: The in_ of this FilterItemsRequest.
        """
        self._in_ = in_
