from typing import List, Optional, Type, TypeVar

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.pull_request_with import PullRequestWith


ForSetLike = TypeVar("ForSetLike", bound=Model)


class RepositoryGroupsMixin:
    """ForSet mixin to add support for `repositories` and `repogroups`."""

    @property
    def repositories(self) -> List[str]:
        """Gets the repositories of this ForSet.

        :return: The repositories of this ForSet.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: List[str]):
        """Sets the repositories of this ForSet.

        :param repositories: The repositories of this ForSet.
        """
        if repositories is None:
            raise ValueError("Invalid value for `repositories`, must not be `None`")
        if len(repositories) == 0:
            raise ValueError("Invalid value for `repositories`, must not be an empty list")
        if self._repogroups is not None:
            for i, group in enumerate(self._repogroups):
                for j, v in enumerate(group):
                    if v >= len(repositories):
                        raise ValueError(
                            "`repogroups[%d][%d]` = %s must be less than the number of "
                            "repositories (%d)" % (i, j, v, len(repositories)))

        self._repositories = repositories

    @property
    def repogroups(self) -> Optional[List[List[int]]]:
        """Gets the repogroups of this ForSet.

        :return: The repogroups of this ForSet.
        """
        return self._repogroups

    @repogroups.setter
    def repogroups(self, repogroups: Optional[List[List[int]]]):
        """Sets the repogroups of this ForSet.

        :param repogroups: The repogroups of this ForSet.
        """
        if repogroups is not None:
            if len(repogroups) == 0:
                raise ValueError("`repogroups` must contain at least one list")
            for i, group in enumerate(repogroups):
                if len(group) == 0:
                    raise ValueError("`repogroups[%d]` must contain at least one element" % i)
                for j, v in enumerate(group):
                    if v < 0:
                        raise ValueError(
                            "`repogroups[%d][%d]` = %s must not be negative" % (i, j, v))
                    if self._repositories is not None and v >= len(self._repositories):
                        raise ValueError(
                            "`repogroups[%d][%d]` = %s must be less than the number of "
                            "repositories (%d)" % (i, j, v, len(self._repositories)))
                if len(set(group)) < len(group):
                    raise ValueError("`repogroups[%d]` has duplicate items" % i)

        self._repogroups = repogroups

    def select_repogroup(self: ForSetLike, index: int) -> ForSetLike:
        """Change `repositories` to point at the specified group and clear `repogroups`."""
        fs = self.copy()
        if self.repogroups is None:
            if index > 0:
                raise IndexError("%d is out of range (no repogroups)" % index)
            return fs
        if index >= len(self.repogroups):
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.repogroups)))
        fs.repogroups = None
        fs.repositories = [self.repositories[i] for i in self.repogroups[index]]
        return fs


def make_common_pull_request_filters(prefix_labels: str) -> Type[Model]:
    """Generate CommonPullRequestFilters class with the specified label properties name prefix."""

    class CommonPullRequestFilters(Model):
        """A few filters that are specific to filtering PR-related entities."""

        openapi_types = {
            prefix_labels + "labels_include": Optional[List[str]],
            prefix_labels + "labels_exclude": Optional[List[str]],
            "jira": Optional[JIRAFilter],
        }

        attribute_map = {
            prefix_labels + "labels_include": prefix_labels + "labels_include",
            prefix_labels + "labels_exclude": prefix_labels + "labels_exclude",
            "jira": "jira",
        }

        __enable_slots__ = False

        def __init__(self, **kwargs):
            """Will be overwritten later."""
            setattr(self, li_name := f"_{prefix_labels}labels_include", kwargs.get(li_name[1:]))
            setattr(self, le_name := f"_{prefix_labels}labels_exclude", kwargs.get(le_name[1:]))
            self._jira = kwargs.get("jira")

        def _get_labels_include(self) -> Optional[List[str]]:
            """Gets the labels_include of this CommonPullRequestFilters.

            :return: The labels_include of this CommonPullRequestFilters.
            """
            return getattr(self, f"_{prefix_labels}labels_include")

        def _set_labels_include(self, labels_include: Optional[List[str]]):
            """Sets the labels_include of this CommonPullRequestFilters.

            :param labels_include: The labels_include of this CommonPullRequestFilters.
            """
            setattr(self, f"_{prefix_labels}labels_include", labels_include)

        def _get_labels_exclude(self) -> Optional[List[str]]:
            """Gets the labels_exclude of this CommonPullRequestFilters.

            :return: The labels_exclude of this CommonPullRequestFilters.
            """
            return getattr(self, f"_{prefix_labels}labels_exclude")

        def _set_labels_exclude(self, labels_exclude: Optional[List[str]]):
            """Sets the labels_exclude of this CommonPullRequestFilters.

            :param labels_exclude: The labels_exclude of this CommonPullRequestFilters.
            """
            setattr(self, f"_{prefix_labels}labels_exclude", labels_exclude)

        @property
        def jira(self) -> Optional[JIRAFilter]:
            """Gets the jira of this CommonPullRequestFilters.

            :return: The jira of this CommonPullRequestFilters.
            """
            return self._jira

        @jira.setter
        def jira(self, jira: Optional[JIRAFilter]):
            """Sets the jira of this CommonPullRequestFilters.

            :param jira: The jira of this CommonPullRequestFilters.
            """
            self._jira = jira

    # we cannot do this at once because it crashes the ast module
    CommonPullRequestFilters.__init__.__doc__ = f"""
    Initialize a new instance of CommonPullRequestFilters.

    :param {prefix_labels}labels_include: The labels_include of this CommonPullRequestFilters.
    :param {prefix_labels}labels_exclude: The labels_exclude of this CommonPullRequestFilters.
    :param jira: The jira of this CommonPullRequestFilters.
    """

    setattr(CommonPullRequestFilters, prefix_labels + "labels_include", property(
        CommonPullRequestFilters._get_labels_include,
        CommonPullRequestFilters._set_labels_include,
    ))
    setattr(CommonPullRequestFilters, prefix_labels + "labels_exclude", property(
        CommonPullRequestFilters._get_labels_exclude,
        CommonPullRequestFilters._set_labels_exclude,
    ))

    return CommonPullRequestFilters


CommonPullRequestFilters = make_common_pull_request_filters("")


class _ForSet(Model, RepositoryGroupsMixin):
    """This class is auto generated by OpenAPI Generator (https://openapi-generator.tech)."""

    openapi_types = {
        "repositories": List[str],
        "repogroups": Optional[List[List[int]]],
        "with_": Optional[PullRequestWith],
        "withgroups": Optional[List[PullRequestWith]],
        "lines": Optional[List[int]],
        "environments": Optional[List[str]],
    }

    attribute_map = {
        "repositories": "repositories",
        "repogroups": "repogroups",
        "with_": "with",
        "withgroups": "withgroups",
        "lines": "lines",
        "environments": "environments",
    }

    __enable_slots__ = False

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
        repogroups: Optional[List[List[int]]] = None,
        with_: Optional[PullRequestWith] = None,
        withgroups: Optional[List[PullRequestWith]] = None,
        lines: Optional[List[int]] = None,
        environments: Optional[List[str]] = None,
    ):
        """ForSet - a model defined in OpenAPI

        :param repositories: The repositories of this ForSet.
        :param repogroups: The repogroups of this ForSet.
        :param with_: The with of this ForSet.
        :param withgroups: The withgroups of this ForSet.
        :param lines: The lines of this ForSet.
        :param environments: The environments of this ForSet.
        """
        self._repositories = repositories
        self._repogroups = repogroups
        self._with_ = with_
        self._withgroups = withgroups
        self._lines = lines
        self._environments = environments

    @property
    def with_(self) -> Optional[PullRequestWith]:
        """Gets the with_ of this PullRequest.

        List of developers related to this PR.

        :return: The with_ of this PullRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[PullRequestWith]):
        """Sets the with_ of this PullRequest.

        List of developers related to this PR.

        :param with_: The with_ of this PullRequest.
        """
        self._with_ = with_

    @property
    def withgroups(self) -> Optional[List[PullRequestWith]]:
        """Gets the withgroups of this PullRequest.

        List of developers related to this PR.

        :return: The withgroups of this PullRequest.
        """
        return self._withgroups

    @withgroups.setter
    def withgroups(self, withgroups: Optional[List[PullRequestWith]]):
        """Sets the withgroups of this PullRequest.

        List of developers related to this PR.

        :param withgroups: The withgroups of this PullRequest.
        """
        self._withgroups = withgroups

    @property
    def lines(self) -> Optional[List[int]]:
        """Gets the lines of this ForSet.

        :return: The lines of this ForSet.
        """
        return self._lines

    @lines.setter
    def lines(self, lines: Optional[List[int]]):
        """Sets the lines of this ForSet.

        :param lines: The lines of this ForSet.
        """
        if lines is not None:
            if len(lines) < 2:
                raise ValueError("`lines` must contain at least 2 elements")
            if lines[0] < 0:
                raise ValueError("all elements of `lines` must be non-negative")
            for i, val in enumerate(lines[:-1]):
                if val >= lines[i + 1]:
                    raise ValueError("`lines` must monotonically increase")
        self._lines = lines

    @property
    def environments(self) -> Optional[List[int]]:
        """Gets the environments of this ForSet.

        :return: The environments of this ForSet.
        """
        return self._environments

    @environments.setter
    def environments(self, environments: Optional[List[int]]):
        """Sets the environments of this ForSet.

        :param environments: The environments of this ForSet.
        """
        self._environments = environments

    def select_lines(self, index: int) -> "ForSet":
        """Change `lines` to point at the specified line range."""
        fs = self.copy()
        if self.lines is None:
            if index > 0:
                raise IndexError("%d is out of range (no lines)" % index)
            return fs
        if index >= len(self.lines) - 1:
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.lines) - 1))
        fs.lines = [fs.lines[index], fs.lines[index + 1]]
        return fs

    def select_withgroup(self, index: int) -> "ForSet":
        """Change `with` to point at the specified `withgroup`."""
        fs = self.copy()
        if self.withgroups is None:
            if index > 0:
                raise IndexError("%d is out of range (no withgroups)" % index)
            return fs
        if index >= len(self.withgroups):
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.withgroups) - 1))
        fs.with_ = self.withgroups[index]
        fs.withgroups = None
        return fs


ForSet = AllOf(_ForSet, CommonPullRequestFilters, name="ForSet", module=__name__)
