from typing import Optional

from email_validator import validate_email

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.invitation_link import _InvitationLink


class _AcceptedInvitation(Model):
    openapi_types = {"name": str, "email": str}
    attribute_map = {"name": "name", "email": "email"}

    def __init__(self,
                 name: Optional[str] = None,
                 email: Optional[str] = None):
        """AcceptedInvitation - a model defined in OpenAPI

        :param name: The name of this AcceptedInvitation.
        :param email: The name of this AcceptedInvitation.
        """
        self._name = name
        self._email = email

    @property
    def name(self) -> Optional[str]:
        """Gets the name of this AcceptedInvitation.

        Calling user's name.

        :return: The name of this AcceptedInvitation.
        """
        return self._name

    @name.setter
    def name(self, name: Optional[str]):
        """Sets the name of this AcceptedInvitation.

        Calling user's name.

        :param name: The name of this AcceptedInvitation.
        """
        self._name = name

    @property
    def email(self) -> Optional[str]:
        """Gets the email of this AcceptedInvitation.

        Calling user's email.

        :return: The email of this AcceptedInvitation.
        """
        return self._email

    @email.setter
    def email(self, email: Optional[str]):
        """Sets the email of this AcceptedInvitation.

        Calling user's email.

        :param email: The email of this AcceptedInvitation.
        """
        if email is not None:
            email = validate_email(email, check_deliverability=False).email
        self._email = email


AcceptedInvitation = AllOf(_AcceptedInvitation,
                           _InvitationLink,
                           name="AcceptedInvitation",
                           module=__name__)
