from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_pipeline_stage import PullRequestPipelineStage
from athenian.api.models.web.pull_request_property import PullRequestProperty
from athenian.api.models.web.pull_request_with import PullRequestWith


class FilterPullRequestsRequest(Model):
    """PR filters for /filter/pull_requests."""

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        in_: Optional[List[str]] = None,
        stages: Optional[List[str]] = None,
        properties: Optional[List[str]] = None,
        with_: Optional[PullRequestWith] = None,
    ):
        """FilterPullRequestsRequest - a model defined in OpenAPI

        :param account: The account of this FilterPullRequestsRequest.
        :param date_from: The date_from of this FilterPullRequestsRequest.
        :param date_to: The date_to of this FilterPullRequestsRequest.
        :param in_: The in_ of this FilterPullRequestsRequest.
        :param stages: The properties of this FilterPullRequestsRequest.
        :param properties: The properties of this FilterPullRequestsRequest.
        :param with_: The with_ of this FilterPullRequestsRequest.
        """
        self.openapi_types = {
            "account": int,
            "date_from": date,
            "date_to": date,
            "in_": List[str],
            "stages": List[str],
            "properties": List[str],
            "with_": PullRequestWith,
        }

        self.attribute_map = {
            "account": "account",
            "date_from": "date_from",
            "date_to": "date_to",
            "in_": "in",
            "stages": "stages",
            "properties": "properties",
            "with_": "with",
        }

        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._in_ = in_
        self._stages = stages
        self._properties = properties
        self._with_ = with_

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
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterPullRequestsRequest.

        :return: The in_ of this FilterPullRequestsRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterPullRequestsRequest.

        :param in_: The in_ of this FilterPullRequestsRequest.
        """
        self._in_ = in_

    @property
    def stages(self) -> List[str]:
        """Gets the properties of this FilterPullRequestsRequest.

        :return: The properties of this FilterPullRequestsRequest.
        """
        return self._stages

    @stages.setter
    def stages(self, stages: List[str]):
        """Sets the properties of this FilterPullRequestsRequest.

        :param stages: The properties of this FilterPullRequestsRequest.
        """
        for stage in stages:
            if stage not in PullRequestPipelineStage.ALL:
                raise ValueError("Invalid stage: %s" % stage)

        self._stages = stages

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
        for stage in properties:
            if stage not in PullRequestProperty.ALL:
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
