from typing import Optional

from athenian.api.models.web.account import _Account
from athenian.api.models.web.base_model_ import AllOf, Model


class _LogicalRepositoryGetRequest(Model, sealed=False):
    attribute_types = {"name": str}
    attribute_map = {"name": "name"}

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """ReleaseMatchRequest - a model defined in OpenAPI

        :param name: The name of this Request.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Gets the name of this Request.

        :return: The name of this Request.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Request.

        :param name: The name of this Request.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name


LogicalRepositoryGetRequest = AllOf(
    _LogicalRepositoryGetRequest, _Account, name="LogicalRepositoryGetRequest", module=__name__
)
