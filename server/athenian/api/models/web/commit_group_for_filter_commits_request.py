import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.commit import Commit


class CommitGroupForFilterCommitsRequest(Model):
    """List of commits that were committed starting from a specific time and ending at \
    +granularity."""

    def __init__(self,
                 date: Optional[datetime.date] = None,
                 commits: Optional[List[Commit]] = None):
        """CommitGroupForFilterCommitsRequest - a model defined in OpenAPI

        :param date: The date of this CommitGroupForFilterCommitsRequest.
        :param commits: The commits of this CommitGroupForFilterCommitsRequest.
        """
        self.openapi_types = {"date": datetime.date, "commits": List[Commit]}

        self.attribute_map = {"date": "date", "commits": "commits"}

        self._date = date
        self._commits = commits

    @property
    def date(self) -> datetime.date:
        """Gets the date of this CommitGroupForFilterCommitsRequest.

        Commits were pushed beginning with this date. They were not pushed later than +granularity.

        :return: The date of this CommitGroupForFilterCommitsRequest.
        """
        return self._date

    @date.setter
    def date(self, date: datetime.date):
        """Sets the date of this CommitGroupForFilterCommitsRequest.

        Commits were pushed beginning with this date. They were not pushed later than +granularity.

        :param date: The date of this CommitGroupForFilterCommitsRequest.
        """
        if date is None:
            raise ValueError("Invalid value for `date`, must not be `None`")

        self._date = date

    @property
    def commits(self) -> List[Commit]:
        """Gets the commits of this CommitGroupForFilterCommitsRequest.

        :return: The commits of this CommitGroupForFilterCommitsRequest.
        """
        return self._commits

    @commits.setter
    def commits(self, commits: List[Commit]):
        """Sets the commits of this CommitGroupForFilterCommitsRequest.

        :param commits: The commits of this CommitGroupForFilterCommitsRequest.
        """
        if commits is None:
            raise ValueError("Invalid value for `commits`, must not be `None`")

        self._commits = commits
