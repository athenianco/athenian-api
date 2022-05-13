from typing import Optional

from athenian.api.models.web.base_model_ import Model


class InvitationCheckResult(Model):
    """Result of checking an invitation URL: invitation type, whether it is correctly formed \
    and is enabled."""

    INVITATION_TYPE_ADMIN = "admin"
    INVITATION_TYPE_REGULAR = "regular"
    attribute_types = {"active": Optional[bool], "type": Optional[str], "valid": bool}
    attribute_map = {"active": "active", "type": "type", "valid": "valid"}

    def __init__(self, active: Optional[bool] = None, type: Optional[str] = None,
                 valid: Optional[bool] = None):
        """InvitationCheckResult - a model defined in OpenAPI

        :param active: The active of this InvitationCheckResult.
        :param type: The type of this InvitationCheckResult.
        :param valid: The valid of this InvitationCheckResult.
        """
        self._active = active
        self._type = type
        self._valid = valid

    @property
    def active(self) -> Optional[bool]:
        """Gets the active of this InvitationCheckResult.

        Value indicating whether the invitation is still enabled.

        :return: The active of this InvitationCheckResult.
        """
        return self._active

    @active.setter
    def active(self, active: Optional[bool]):
        """Sets the active of this InvitationCheckResult.

        Value indicating whether the invitation is still enabled.

        :param active: The active of this InvitationCheckResult.
        """
        self._active = active

    @property
    def type(self) -> Optional[str]:
        """Gets the type of this InvitationCheckResult.

        Invited user's account membership status.

        :return: The type of this InvitationCheckResult.
        """
        return self._type

    @type.setter
    def type(self, type: Optional[str]):
        """Sets the type of this InvitationCheckResult.

        Invited user's account membership status.

        :param type: The type of this InvitationCheckResult.
        """
        allowed_values = [self.INVITATION_TYPE_ADMIN, self.INVITATION_TYPE_REGULAR, None]
        if type not in allowed_values:
            raise ValueError("Invalid value for `type` %s, must be one of %s" %
                             (type, allowed_values))

        self._type = type

    @property
    def valid(self) -> bool:
        """Gets the valid of this InvitationCheckResult.

        Value indicating whether the invitation URL is correctly formed.

        :return: The valid of this InvitationCheckResult.
        """
        return self._valid

    @valid.setter
    def valid(self, valid: bool):
        """Sets the valid of this InvitationCheckResult.

        Value indicating whether the invitation URL is correctly formed.

        :param valid: The valid of this InvitationCheckResult.
        """
        if valid is None:
            raise ValueError("Invalid value for `valid`, must not be `None`")

        self._valid = valid
