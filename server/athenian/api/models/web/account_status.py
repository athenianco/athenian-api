from typing import Optional

from athenian.api.models.web.base_model_ import Model


class AccountStatus(Model):
    """Status of the user's account membership."""

    attribute_types = {
        "is_admin": bool,
        "expired": bool,
        "has_ci": bool,
        "has_jira": bool,
        "has_deployments": bool,
    }

    attribute_map = {
        "is_admin": "is_admin",
        "expired": "expired",
        "has_ci": "has_ci",
        "has_jira": "has_jira",
        "has_deployments": "has_deployments",
    }

    def __init__(
        self,
        is_admin: Optional[bool] = None,
        expired: Optional[bool] = None,
        has_ci: Optional[bool] = None,
        has_jira: Optional[bool] = None,
        has_deployments: Optional[bool] = None,
    ):
        """AccountStatus - a model defined in OpenAPI

        :param is_admin: The is_admin of this AccountStatus.
        :param expired: The expired of this AccountStatus.
        :param has_ci: The has_ci of this AccountStatus.
        :param has_jira: The has_jira of this AccountStatus.
        :param has_deployments: The has_deployments of this AccountStatus.
        """
        self._is_admin = is_admin
        self._expired = expired
        self._has_ci = has_ci
        self._has_jira = has_jira
        self._has_deployments = has_deployments

    @property
    def is_admin(self) -> bool:
        """Gets the is_admin of this AccountStatus.

        Indicates whether the user is an account administrator.

        :return: The is_admin of this AccountStatus.
        """
        return self._is_admin

    @is_admin.setter
    def is_admin(self, is_admin: bool):
        """Sets the is_admin of this AccountStatus.

        Indicates whether the user is an account administrator.

        :param is_admin: The is_admin of this AccountStatus.
        """
        if is_admin is None:
            raise ValueError("Invalid value for `is_admin`, must not be `None`")

        self._is_admin = is_admin

    @property
    def expired(self) -> bool:
        """Gets the expired of this AccountStatus.

        Indicates whether the account is disabled.

        :return: The expired of this AccountStatus.
        """
        return self._expired

    @expired.setter
    def expired(self, expired: bool):
        """Sets the expired of this AccountStatus.

        Indicates whether the account is disabled.

        :param expired: The expired of this AccountStatus.
        """
        if expired is None:
            raise ValueError("Invalid value for `expired`, must not be `None`")

        self._expired = expired

    @property
    def has_ci(self) -> bool:
        """Gets the has_ci of this AccountStatus.

        Indicates whether the account permitted the access to check suites.

        :return: The has_ci of this AccountStatus.
        """
        return self._has_ci

    @has_ci.setter
    def has_ci(self, has_ci: bool):
        """Sets the has_ci of this AccountStatus.

        Indicates whether the account permitted the access to check suites.

        :param has_ci: The has_ci of this AccountStatus.
        """
        if has_ci is None:
            raise ValueError("Invalid value for `has_ci`, must not be `None`")

        self._has_ci = has_ci

    @property
    def has_jira(self) -> bool:
        """Gets the has_jira of this AccountStatus.

        Indicates whether the account installed the integration with JIRA.

        :return: The has_jira of this AccountStatus.
        """
        return self._has_jira

    @has_jira.setter
    def has_jira(self, has_jira: bool):
        """Sets the has_jira of this AccountStatus.

        Indicates whether the account installed the integration with JIRA.

        :param has_jira: The has_jira of this AccountStatus.
        """
        if has_jira is None:
            raise ValueError("Invalid value for `has_jira`, must not be `None`")

        self._has_jira = has_jira

    @property
    def has_deployments(self) -> bool:
        """Gets the has_deployments of this AccountStatus.

        Indicates whether the account has submitted at least one deployment.

        :return: The has_deployments of this AccountStatus.
        """
        return self._has_deployments

    @has_deployments.setter
    def has_deployments(self, has_deployments: bool):
        """Sets the has_deployments of this AccountStatus.

        Indicates whether the account has submitted at least one deployment.

        :param has_deployments: The has_deployments of this AccountStatus.
        """
        if has_deployments is None:
            raise ValueError("Invalid value for `has_deployments`, must not be `None`")

        self._has_deployments = has_deployments
