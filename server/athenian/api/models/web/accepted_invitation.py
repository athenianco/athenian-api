from typing import Optional

from email_validator import validate_email

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.invitation_link import _InvitationLink


class _AcceptedInvitation(Model):
    name: Optional[str]
    email: Optional[str]

    def validate_email(self, email: Optional[str]) -> Optional[str]:
        """Sets the email of this AcceptedInvitation.

        Calling user's email.

        :param email: The email of this AcceptedInvitation.
        """
        if email is not None:
            email = validate_email(email, check_deliverability=False).email
        return email


AcceptedInvitation = AllOf(
    _AcceptedInvitation, _InvitationLink, name="AcceptedInvitation", module=__name__,
)
