from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAUser(Model):
    """Details about a JIRA user."""

    name: str
    avatar: str
    type: str
    developer: Optional[str]

    def validate_type(self, type: str) -> str:
        """Sets the type of this JIRAUser.

        * `atlassian` indicates a regular account backed by a human.
        * `app` indicates a service account.
        * `customer` indicates an external service desk account.

        :param type: The type of this JIRAUser.
        """
        allowed_values = {"atlassian", "app", "customer"}
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` (%s), must be one of %s" % (type, allowed_values),
            )

        return type
