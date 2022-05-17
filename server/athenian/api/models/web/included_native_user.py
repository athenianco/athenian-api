from typing import Optional

from athenian.api.models.web.base_model_ import Model


class IncludedNativeUser(Model):
    """User traits such as the avatar URL."""

    attribute_types = {"avatar": str}
    attribute_map = {"avatar": "avatar"}

    def __init__(self, avatar: Optional[str] = None):
        """IncludedNativeUser - a model defined in OpenAPI

        :param avatar: The avatar of this IncludedNativeUser.
        """
        self._avatar = avatar

    @property
    def avatar(self) -> str:
        """Gets the avatar of this IncludedNativeUser.

        :return: The avatar of this IncludedNativeUser.
        """
        return self._avatar

    @avatar.setter
    def avatar(self, avatar: str):
        """Sets the avatar of this IncludedNativeUser.

        :param avatar: The avatar of this IncludedNativeUser.
        """
        if avatar is None:
            raise ValueError("Invalid value for `avatar`, must not be `None`")

        self._avatar = avatar
