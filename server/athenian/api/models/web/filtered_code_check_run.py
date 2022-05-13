from datetime import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.code_check_run_statistics import CodeCheckRunStatistics


class FilteredCodeCheckRun(Model):
    """Mined information about a code check run."""

    attribute_types = {
        "title": str,
        "repository": str,
        "last_execution_time": datetime,
        "last_execution_url": str,
        "total_stats": CodeCheckRunStatistics,
        "prs_stats": CodeCheckRunStatistics,
        "size_groups": List[int],
    }

    attribute_map = {
        "title": "title",
        "repository": "repository",
        "last_execution_time": "last_execution_time",
        "last_execution_url": "last_execution_url",
        "total_stats": "total_stats",
        "prs_stats": "prs_stats",
        "size_groups": "size_groups",
    }

    def __init__(
        self,
        title: Optional[str] = None,
        repository: Optional[str] = None,
        last_execution_time: Optional[datetime] = None,
        last_execution_url: Optional[str] = None,
        total_stats: Optional[CodeCheckRunStatistics] = None,
        prs_stats: Optional[CodeCheckRunStatistics] = None,
        size_groups: Optional[List[int]] = None,
    ):
        """FilteredCodeCheckRun - a model defined in OpenAPI

        :param title: The title of this FilteredCodeCheckRun.
        :param repository: The repository of this FilteredCodeCheckRun.
        :param last_execution_time: The last_execution_time of this FilteredCodeCheckRun.
        :param last_execution_url: The last_execution_url of this FilteredCodeCheckRun.
        :param total_stats: The total_stats of this FilteredCodeCheckRun.
        :param prs_stats: The prs_stats of this FilteredCodeCheckRun.
        :param size_groups: The size_groups of this FilteredCodeCheckRun.
        """
        self._title = title
        self._repository = repository
        self._last_execution_time = last_execution_time
        self._last_execution_url = last_execution_url
        self._total_stats = total_stats
        self._prs_stats = prs_stats
        self._size_groups = size_groups

    @property
    def title(self) -> str:
        """Gets the title of this FilteredCodeCheckRun.

        Unique name of the check run.

        :return: The title of this FilteredCodeCheckRun.
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this FilteredCodeCheckRun.

        Unique name of the check run.

        :param title: The title of this FilteredCodeCheckRun.
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")

        self._title = title

    @property
    def repository(self) -> str:
        """Gets the repository of this FilteredCodeCheckRun.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :return: The repository of this FilteredCodeCheckRun.
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this FilteredCodeCheckRun.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :param repository: The repository of this FilteredCodeCheckRun.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def last_execution_time(self) -> datetime:
        """Gets the last_execution_time of this FilteredCodeCheckRun.

        Timestamp of when the check run launched last time.

        :return: The last_execution_time of this FilteredCodeCheckRun.
        """
        return self._last_execution_time

    @last_execution_time.setter
    def last_execution_time(self, last_execution_time: datetime):
        """Sets the last_execution_time of this FilteredCodeCheckRun.

        Timestamp of when the check run launched last time.

        :param last_execution_time: The last_execution_time of this FilteredCodeCheckRun.
        """
        if last_execution_time is None:
            raise ValueError("Invalid value for `last_execution_time`, must not be `None`")

        self._last_execution_time = last_execution_time

    @property
    def last_execution_url(self) -> str:
        """Gets the last_execution_url of this FilteredCodeCheckRun.

        Link to the check run that launched the latest.

        :return: The last_execution_url of this FilteredCodeCheckRun.
        """
        return self._last_execution_url

    @last_execution_url.setter
    def last_execution_url(self, last_execution_url: str):
        """Sets the last_execution_url of this FilteredCodeCheckRun.

        Link to the check run that launched the latest.

        :param last_execution_url: The last_execution_url of this FilteredCodeCheckRun.
        """
        if last_execution_url is None:
            raise ValueError("Invalid value for `last_execution_url`, must not be `None`")

        self._last_execution_url = last_execution_url

    @property
    def total_stats(self) -> CodeCheckRunStatistics:
        """Gets the total_stats of this FilteredCodeCheckRun.

        :return: The total_stats of this FilteredCodeCheckRun.
        """
        return self._total_stats

    @total_stats.setter
    def total_stats(self, total_stats: CodeCheckRunStatistics):
        """Sets the total_stats of this FilteredCodeCheckRun.

        :param total_stats: The total_stats of this FilteredCodeCheckRun.
        """
        if total_stats is None:
            raise ValueError("Invalid value for `total_stats`, must not be `None`")

        self._total_stats = total_stats

    @property
    def prs_stats(self) -> CodeCheckRunStatistics:
        """Gets the prs_stats of this FilteredCodeCheckRun.

        :return: The prs_stats of this FilteredCodeCheckRun.
        """
        return self._prs_stats

    @prs_stats.setter
    def prs_stats(self, prs_stats: CodeCheckRunStatistics):
        """Sets the prs_stats of this FilteredCodeCheckRun.

        :param prs_stats: The prs_stats of this FilteredCodeCheckRun.
        """
        if prs_stats is None:
            raise ValueError("Invalid value for `prs_stats`, must not be `None`")

        self._prs_stats = prs_stats

    @property
    def size_groups(self) -> List[int]:
        """Gets the size_groups of this FilteredCodeCheckRun.

        Check suite sizes this check run belongs to.

        :return: The size_groups of this FilteredCodeCheckRun.
        """
        return self._size_groups

    @size_groups.setter
    def size_groups(self, size_groups: List[int]):
        """Sets the size_groups of this FilteredCodeCheckRun.

        Check suite sizes this check run belongs to.

        :param size_groups: The size_groups of this FilteredCodeCheckRun.
        """
        if size_groups is None:
            raise ValueError("Invalid value for `size_groups`, must not be `None`")

        self._size_groups = size_groups
