from athenian.api.models.web.base_model_ import Enum, Model


class UserChangeStatus(metaclass=Enum):
    """Options for changing the user account membership."""

    REGULAR = "regular"
    ADMIN = "admin"
    BANISHED = "banished"


class AccountUserChangeRequest(Model):
    """Request to change an account member's status."""

    account: int
    user: str
    status: str

    def validate_status(self, status: str) -> str:
        """Sets the status of this AccountUserChangeRequest.

        Account membership role.

        :param status: The status of this AccountUserChangeRequest.
        """
        if status not in UserChangeStatus:
            raise ValueError(
                "Invalid value for `status` (%s), must be one of %s"
                % (status, list(UserChangeStatus)),
            )

        return status
