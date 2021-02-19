from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_notification import ReleaseNotification


class NotifyReleaseRequest(Model):
    """Request body of `/notify/release`."""

    openapi_types = {
        "account": int,
        "notifications": List[ReleaseNotification],
    }

    attribute_map = {"account": "account", "notifications": "notifications"}

    def __init__(self,
                 account: Optional[int] = None,
                 notifications: Optional[List[ReleaseNotification]] = None,
                 ):
        """NotifyReleaseRequest - a model defined in OpenAPI

        :param account: The account of this NotifyReleaseRequest.
        :param notifications: The notifications of this NotifyReleaseRequest.
        """
        self._account = account
        self._notifications = notifications

    @property
    def account(self) -> int:
        """Gets the account of this NotifyReleaseRequest.

        Account ID.

        :return: The account of this NotifyReleaseRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this NotifyReleaseRequest.

        Account ID.

        :param account: The account of this NotifyReleaseRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def notifications(self) -> List[ReleaseNotification]:
        """Gets the notifications of this NotifyReleaseRequest.

        :return: The notifications of this NotifyReleaseRequest.
        """
        return self._notifications

    @notifications.setter
    def notifications(self, notifications: List[ReleaseNotification]):
        """Sets the notifications of this NotifyReleaseRequest.

        :param notifications: The notifications of this NotifyReleaseRequest.
        """
        if notifications is None:
            raise ValueError("Invalid value for `notifications`, must not be `None`")

        self._notifications = notifications
