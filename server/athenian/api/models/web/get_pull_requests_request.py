from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_numbers import PullRequestNumbers


class GetPullRequestsRequest(Model):
    """Request body of `/get/pull_requests`. Declaration of which PRs the user wants to analyze."""

    attribute_types = {
        "account": int,
        "prs": List[PullRequestNumbers],
        "environment": Optional[str],
    }
    attribute_map = {
        "account": "account",
        "prs": "prs",
        "environment": "environment",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        prs: Optional[List[PullRequestNumbers]] = None,
        environment: Optional[str] = None,
    ):
        """GetPullRequestsRequest - a model defined in OpenAPI

        :param account: The account of this GetPullRequestsRequest.
        :param prs: The prs of this GetPullRequestsRequest.
        :param environment: The environment of this GetPullRequestsRequest.
        """
        self._account = account
        self._prs = prs
        self._environment = environment

    @property
    def account(self) -> int:
        """Gets the account of this GetPullRequestsRequest.

        Account ID.

        :return: The account of this GetPullRequestsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this GetPullRequestsRequest.

        Account ID.

        :param account: The account of this GetPullRequestsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def prs(self) -> List[PullRequestNumbers]:
        """Gets the prs of this GetPullRequestsRequest.

        List of repositories and PR numbers to _analyze.

        :return: The prs of this GetPullRequestsRequest.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: List[PullRequestNumbers]):
        """Sets the prs of this GetPullRequestsRequest.

        List of repositories and PR numbers to _analyze.

        :param prs: The prs of this GetPullRequestsRequest.
        """
        if prs is None:
            raise ValueError("Invalid value for `prs`, must not be `None`")

        self._prs = prs

    @property
    def environment(self) -> Optional[str]:
        """Gets the environment of this GetPullRequestsRequest.

        Deployment environment to apply.

        :return: The environment of this GetPullRequestsRequest.
        """
        return self._environment

    @environment.setter
    def environment(self, environment: Optional[str]):
        """Sets the environment of this GetPullRequestsRequest.

        Deployment environment to apply.

        :param environment: The environment of this GetPullRequestsRequest.
        """
        self._environment = environment
