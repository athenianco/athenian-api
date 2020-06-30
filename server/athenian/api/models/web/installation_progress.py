from datetime import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.table_fetching_progress import TableFetchingProgress


class InstallationProgress(Model):
    """Data fetching progress of the Athenian metadata retrieval app."""

    openapi_types = {
        "started_date": datetime,
        "finished_date": datetime,
        "owner": str,
        "repositories": int,
        "tables": List[TableFetchingProgress],
    }

    attribute_map = {
        "started_date": "started_date",
        "finished_date": "finished_date",
        "owner": "owner",
        "repositories": "repositories",
        "tables": "tables",
    }

    __slots__ = ["_" + k for k in openapi_types]

    def __init__(
        self,
        started_date: Optional[datetime] = None,
        finished_date: Optional[datetime] = None,
        owner: Optional[str] = None,
        repositories: Optional[int] = None,
        tables: Optional[List[TableFetchingProgress]] = None,
    ):
        """InstallationProgress - a model defined in OpenAPI

        :param started_date: The started_date of this InstallationProgress.
        :param finished_date: The finished_date of this InstallationProgress.
        :param owner: The owner of this InstallationProgress.
        :param repositories: The repositories of this InstallationProgress.
        :param tables: The tables of this InstallationProgress.
        """
        self._started_date = started_date
        self._finished_date = finished_date
        self._owner = owner
        self._repositories = repositories
        self._tables = tables

    @property
    def started_date(self) -> datetime:
        """Gets the started_date of this InstallationProgress.

        Date and time when the historical data collection began.

        :return: The started_date of this InstallationProgress.
        """
        return self._started_date

    @started_date.setter
    def started_date(self, started_date: datetime):
        """Sets the started_date of this InstallationProgress.

        Date and time when the historical data collection began.

        :param started_date: The started_date of this InstallationProgress.
        """
        if started_date is None:
            raise ValueError("Invalid value for `started_date`, must not be `None`")

        self._started_date = started_date

    @property
    def finished_date(self) -> Optional[datetime]:
        """Gets the finished_date of this InstallationProgress.

        Date and time when the historical data collection ended.

        :return: The finished_date of this InstallationProgress.
        """
        return self._finished_date

    @finished_date.setter
    def finished_date(self, finished_date: Optional[datetime]):
        """Sets the finished_date of this InstallationProgress.

        Date and time when the historical data collection ended.

        :param finished_date: The finished_date of this InstallationProgress.
        """
        self._finished_date = finished_date

    @property
    def owner(self) -> str:
        """Gets the owner of this InstallationProgress.

        Native user ID of the person who installed the metadata retrieval app.

        :return: The owner of this InstallationProgress.
        """
        return self._owner

    @owner.setter
    def owner(self, owner: str):
        """Sets the owner of this InstallationProgress.

        Native user ID of the person who installed the metadata retrieval app.

        :param owner: The owner of this InstallationProgress.
        """
        if owner is None:
            raise ValueError("Invalid value for `owner`, must not be `None`")

        self._owner = owner

    @property
    def repositories(self) -> int:
        """Gets the repositories of this InstallationProgress.

        Number of discovered repositories.

        :return: The repositories of this InstallationProgress.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: int):
        """Sets the repositories of this InstallationProgress.

        Number of discovered repositories.

        :param repositories: The repositories of this InstallationProgress.
        """
        if repositories is None:
            raise ValueError("Invalid value for `repositories`, must not be `None`")

        self._repositories = repositories

    @property
    def tables(self) -> List[TableFetchingProgress]:
        """Gets the tables of this InstallationProgress.

        :return: The tables of this InstallationProgress.
        """
        return self._tables

    @tables.setter
    def tables(self, tables: List[TableFetchingProgress]):
        """Sets the tables of this InstallationProgress.

        :param tables: The tables of this InstallationProgress.
        """
        if tables is None:
            raise ValueError("Invalid value for `tables`, must not be `None`")

        self._tables = tables
