from typing import Optional

from athenian.api.models.web.base_model_ import Model


class PullRequestSetIncludeUser(Model):
    """User traits such as the avatar URL."""

    def __init__(self, avatar: Optional[str] = None):
        """PullRequestSetIncludeUsers - a model defined in OpenAPI

        :param avatar: The avatar of this PullRequestSetIncludeUsers.
        """
        self.openapi_types = {"avatar": str}

        self.attribute_map = {"avatar": "avatar"}

        self._avatar = avatar

    @property
    def avatar(self) -> str:
        """Gets the avatar of this PullRequestSetIncludeUsers.

        :return: The avatar of this PullRequestSetIncludeUsers.
        """
        return self._avatar

    @avatar.setter
    def avatar(self, avatar: str):
        """Sets the avatar of this PullRequestSetIncludeUsers.

        :param avatar: The avatar of this PullRequestSetIncludeUsers.
        """
        if avatar is None:
            raise ValueError("Invalid value for `avatar`, must not be `None`")

        self._avatar = avatar
