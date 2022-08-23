from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.goal_template import GoalTemplateCommon


class _GoalTemplateCreateRequest(Model, sealed=False):
    """Goal Template creation request."""

    attribute_types = {"account": int}

    def __init__(self, account: Optional[int] = None):
        """GoalTemplateCreateRequest - a model defined in OpenAPI

        :param account: The account of this GoalTemplateCreateRequest.
        """
        self._account = account

    @property
    def account(self) -> int:
        """Gets the account of this GoalTemplateCreateRequest.

        Account identifier. That account will own the created goal template.
        """
        return self._account

    @account.setter
    def account(self, account: int) -> None:
        """Sets the account of this GoalTemplateCreateRequest."""
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account


GoalTemplateCreateRequest = AllOf(
    GoalTemplateCommon,
    _GoalTemplateCreateRequest,
    name="GoalTemplateCreateRequest",
    module=__name__,
)
