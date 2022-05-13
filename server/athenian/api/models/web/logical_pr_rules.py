from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class LogicalPRRules(Model):
    """Rules to match PRs to logical repository."""

    attribute_types = {"title": str, "labels_include": List[str]}

    attribute_map = {"title": "title", "labels_include": "labels_include"}

    def __init__(self,
                 title: Optional[str] = None,
                 labels_include: Optional[List[str]] = None):
        """LogicalPRRules - a model defined in OpenAPI

        :param title: The title of this LogicalPRRules.
        :param labels_include: The labels_include of this LogicalPRRules.
        """
        self._title = title
        self._labels_include = labels_include

    @property
    def title(self) -> Optional[str]:
        """Gets the title of this LogicalPRRules.

        Regular expression to match the PR title. It must be a match starting from the start of
        the string.

        :return: The title of this LogicalPRRules.
        """
        return self._title

    @title.setter
    def title(self, title: Optional[str]):
        """Sets the title of this LogicalPRRules.

        Regular expression to match the PR title. It must be a match starting from the start of
        the string.

        :param title: The title of this LogicalPRRules.
        """
        self._title = title

    @property
    def labels_include(self) -> Optional[List[str]]:
        """Gets the labels_include of this LogicalPRRules.

        Any matching label puts the PR into the logical repository.

        :return: The labels_include of this LogicalPRRules.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: Optional[List[str]]):
        """Sets the labels_include of this LogicalPRRules.

        Any matching label puts the PR into the logical repository.

        :param labels_include: The labels_include of this LogicalPRRules.
        """
        self._labels_include = labels_include
