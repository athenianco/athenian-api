from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.work_type import WorkType


class WorkTypePutRequest(Model):
    """Request body of `PUT /settings/work_type`."""

    attribute_types = {"account": int, "work_type": WorkType}

    attribute_map = {"account": "account", "work_type": "work_type"}

    def __init__(self,
                 account: Optional[int] = None,
                 work_type: Optional[WorkType] = None):
        """WorkTypePutRequest - a model defined in OpenAPI

        :param account: The account of this WorkTypePutRequest.
        :param work_type: The work_type of this WorkTypePutRequest.
        """
        self._account = account
        self._work_type = work_type

    @property
    def account(self) -> int:
        """Gets the account of this WorkTypePutRequest.

        Account ID.

        :return: The account of this WorkTypePutRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this WorkTypePutRequest.

        Account ID.

        :param account: The account of this WorkTypePutRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def work_type(self) -> WorkType:
        """Gets the work_type of this WorkTypePutRequest.

        :return: The work_type of this WorkTypePutRequest.
        """
        return self._work_type

    @work_type.setter
    def work_type(self, work_type: WorkType):
        """Sets the work_type of this WorkTypePutRequest.

        :param work_type: The work_type of this WorkTypePutRequest.
        """
        if work_type is None:
            raise ValueError("Invalid value for `work_type`, must not be `None`")

        self._work_type = work_type
