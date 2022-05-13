from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _CommitFilter(Model, sealed=False):
    """Specific parts of the commit filter."""

    attribute_types = {
        "in_": List[str],
        "with_author": List[str],
        "with_committer": List[str],
        "only_default_branch": Optional[bool],
    }

    attribute_map = {
        "in_": "in",
        "with_author": "with_author",
        "with_committer": "with_committer",
        "only_default_branch": "only_default_branch",
    }

    def __init__(
        self,
        in_: Optional[List[str]] = None,
        with_author: Optional[List[str]] = None,
        with_committer: Optional[List[str]] = None,
        only_default_branch: Optional[bool] = None,
    ):
        """CommitFilter - a model defined in OpenAPI

        :param in_: The in of this CommitFilter.
        :param with_author: The with_author of this CommitFilter.
        :param with_committer: The with_committer of this CommitFilter.
        :param only_default_branch: The only_default_branch of this CommitFilter.
        """
        self._in_ = in_
        self._with_author = with_author
        self._with_committer = with_committer
        self._only_default_branch = only_default_branch

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this CommitFilter.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :return: The in_ of this CommitFilter.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this CommitFilter.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :param in_: The in_ of this CommitFilter.
        """
        if in_ is None:
            raise ValueError("Invalid value for `in`, must not be `None`")

        self._in_ = in_

    @property
    def with_author(self) -> List[str]:
        """Gets the with_author of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :return: The with_author of this CommitFilter.
        """
        return self._with_author

    @with_author.setter
    def with_author(self, with_author: List[str]):
        """Sets the with_author of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :param with_author: The with_author of this CommitFilter.
        """
        self._with_author = with_author

    @property
    def with_committer(self) -> List[str]:
        """Gets the with_committer of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :return: The with_committer of this CommitFilter.
        """
        return self._with_committer

    @with_committer.setter
    def with_committer(self, with_committer: List[str]):
        """Sets the with_committer of this CommitFilter.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :param with_committer: The with_committer of this CommitFilter.
        """
        self._with_committer = with_committer

    @property
    def only_default_branch(self) -> bool:
        """Gets the only_default_branch of this CommitFilter.

        Consider only commits that are accessible from the head of the default branch.
        For example, if your workflow includes PR-ing feature branches, you rebase and merge
        them and don't delete afterward, `bypassing_prs` will report the commits in those
        feature branches unless you set `only_default_branch` to `true`.

        :return: The only_default_branch of this CommitFilter.
        """
        return self._only_default_branch or False  # force True or False

    @only_default_branch.setter
    def only_default_branch(self, only_default_branch: Optional[bool]):
        """Sets the only_default_branch of this CommitFilter.

        Consider only commits that are accessible from the head of the default branch.
        For example, if your workflow includes PR-ing feature branches, you rebase and merge
        them and don't delete afterward, `bypassing_prs` will report the commits in those
        feature branches unless you set `only_default_branch` to `true`.

        :param only_default_branch: The only_default_branch of this CommitFilter.
        """
        self._only_default_branch = only_default_branch


CommitFilter = AllOf(_CommitFilter, CommonFilterProperties, name="CommitFilter", module=__name__)
