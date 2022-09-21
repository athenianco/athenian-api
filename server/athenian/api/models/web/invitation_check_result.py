from typing import Optional

from athenian.api.models.web.base_model_ import Model


class InvitationCheckResult(Model):
    """Result of checking an invitation URL: invitation type, whether it is correctly formed \
    and is enabled."""

    INVITATION_TYPE_ADMIN = "admin"
    INVITATION_TYPE_REGULAR = "regular"

    active: Optional[bool]
    type: Optional[str]
    valid: bool

    def validate_type(self, type: Optional[str]) -> Optional[str]:
        """Sets the type of this InvitationCheckResult.

        Invited user's account membership status.

        :param type: The type of this InvitationCheckResult.
        """
        allowed_values = [self.INVITATION_TYPE_ADMIN, self.INVITATION_TYPE_REGULAR, None]
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` %s, must be one of %s" % (type, allowed_values),
            )

        return type
