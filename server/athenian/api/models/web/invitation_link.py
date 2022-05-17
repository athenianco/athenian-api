from typing import Optional

from athenian.api.models.web.base_model_ import Model


class _InvitationLink(Model, sealed=False):
    attribute_types = {"url": str}
    attribute_map = {"url": "url"}

    def __init__(self, url: Optional[str] = None):
        """InvitationLink - a model defined in OpenAPI

        :param url: The url of this InvitationLink.
        """
        self._url = url

    @property
    def url(self) -> str:
        """Gets the url of this InvitationLink.

        Invitation URL. Users are supposed to click it and become regular account members.

        :return: The url of this InvitationLink.
        """
        return self._url

    @url.setter
    def url(self, url: str):
        """Sets the url of this InvitationLink.

        Invitation URL. Users are supposed to click it and become regular account members.

        :param url: The url of this InvitationLink.
        """
        if url is None:
            raise ValueError("Invalid value for `url`, must not be `None`")

        self._url = url


class InvitationLink(_InvitationLink, sealed=True):
    """Product invitation URL."""
