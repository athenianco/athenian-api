from typing import Optional

from athenian.api.models.web.base_model_ import Model


class TableFetchingProgress(Model):
    """Used in InstallationProgress.tables to indicate how much data has been loaded in each DB \
    table."""

    attribute_types = {"fetched": int, "name": str, "total": int}
    attribute_map = {"fetched": "fetched", "name": "name", "total": "total"}

    def __init__(
        self,
        fetched: Optional[int] = None,
        name: Optional[str] = None,
        total: Optional[int] = None,
    ):
        """TableFetchingProgress - a model defined in OpenAPI

        :param fetched: The fetched of this TableFetchingProgress.
        :param name: The name of this TableFetchingProgress.
        :param total: The total of this TableFetchingProgress.
        """
        self._fetched = fetched
        self._name = name
        self._total = total

    def __lt__(self, other: "TableFetchingProgress") -> bool:
        """Implement "<"."""
        return self.name < other.name

    @property
    def fetched(self) -> int:
        """Gets the fetched of this TableFetchingProgress.

        How many records have been stored in the DB table so far.

        :return: The fetched of this TableFetchingProgress.
        """
        return self._fetched

    @fetched.setter
    def fetched(self, fetched: int):
        """Sets the fetched of this TableFetchingProgress.

        How many records have been stored in the DB table so far.

        :param fetched: The fetched of this TableFetchingProgress.
        """
        if fetched is None:
            raise ValueError("Invalid value for `fetched`, must not be `None`")

        self._fetched = fetched

    @property
    def name(self) -> str:
        """Gets the name of this TableFetchingProgress.

        DB table identifier.

        :return: The name of this TableFetchingProgress.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this TableFetchingProgress.

        DB table identifier.

        :param name: The name of this TableFetchingProgress.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def total(self) -> int:
        """Gets the total of this TableFetchingProgress.

        How many records the DB table is expected to have.

        :return: The total of this TableFetchingProgress.
        """
        return self._total

    @total.setter
    def total(self, total: int):
        """Sets the total of this TableFetchingProgress.

        How many records the DB table is expected to have.

        :param total: The total of this TableFetchingProgress.
        """
        if total is None:
            raise ValueError("Invalid value for `total`, must not be `None`")

        self._total = total
