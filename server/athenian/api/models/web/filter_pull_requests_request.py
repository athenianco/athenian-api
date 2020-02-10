from datetime import date
from typing import List, Optional

from athenian.api import serialization
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_with import PullRequestWith


class FilterPullRequestsRequest(Model):
    """PR filters for /filter/pull_requests."""

    def _in_it__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        _in: Optional[List[str]] = None,
        stages: Optional[List[str]] = None,
        _with: Optional[PullRequestWith] = None,
    ):
        """FilterPullRequestsRequest - a model defined in OpenAPI

        :param account: The account of this FilterPullRequestsRequest.
        :param date_from: The date_from of this FilterPullRequestsRequest.
        :param date_to: The date_to of this FilterPullRequestsRequest.
        :param _in: The _in of this FilterPullRequestsRequest.
        :param stages: The stages of this FilterPullRequestsRequest.
        :param _with: The _with of this FilterPullRequestsRequest.
        """
        self.openapi_types = {
            "account": int,
            "date_from": date,
            "date_to": date,
            "_in": List[str],
            "stages": List[str],
            "_with": PullRequestWith,
        }

        self.attribute_map = {
            "account": "account",
            "date_from": "date_from",
            "date_to": "date_to",
            "_in": "in",
            "stages": "stages",
            "_with": "with",
        }

        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._in_ = _in
        self._stages = stages
        self._with_ = _with

    @classmethod
    def from_dict(cls, dikt: dict) -> "FilterPullRequestsRequest":
        """Returns the dict as a model

        :param dikt: A dict.
        :return: The FilterPullRequestsRequest of this FilterPullRequestsRequest.
        """
        return serialization.deserialize_model(dikt, cls)

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
    def _in(self) -> List[str]:
        """Gets the _in of this FilterPullRequestsRequest.

        :return: The _in of this FilterPullRequestsRequest.
        """
        return self._in_

    @_in.setter
    def _in(self, _in: List[str]):
        """Sets the _in of this FilterPullRequestsRequest.

        :param _in: The _in of this FilterPullRequestsRequest.
        """
        self._in_ = _in

    @property
    def stages(self) -> List[str]:
        """Gets the stages of this FilterPullRequestsRequest.

        :return: The stages of this FilterPullRequestsRequest.
        """
        return self._stages

    @stages.setter
    def stages(self, stages: List[str]):
        """Sets the stages of this FilterPullRequestsRequest.

        :param stages: The stages of this FilterPullRequestsRequest.
        """
        self._stages = stages

    @property
    def _with(self) -> Optional[PullRequestWith]:
        """Gets the _with of this FilterPullRequestsRequest.

        :return: The _with of this FilterPullRequestsRequest.
        """
        return self._with_

    @_with.setter
    def _with(self, _with: PullRequestWith):
        """Sets the _with of this FilterPullRequestsRequest.

        :param _with: The _with of this FilterPullRequestsRequest.
        """
        self._with_ = _with
