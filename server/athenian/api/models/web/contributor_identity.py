from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class ContributorIdentity(Model):
    """Information about a contributor that may be utilized to match identities."""

    attribute_types = {"emails": Optional[List[str]], "names": Optional[List[str]]}

    attribute_map = {"emails": "emails", "names": "names"}

    def __init__(self,
                 emails: Optional[List[str]] = None,
                 names: Optional[List[str]] = None):
        """ContributorIdentity - a model defined in OpenAPI

        :param emails: The emails of this ContributorIdentity.
        :param names: The names of this ContributorIdentity.
        """
        self._emails = emails
        self._names = names

    @property
    def emails(self) -> Optional[List[str]]:
        """Gets the emails of this ContributorIdentity.

        Email addresses belonging to the person.

        :return: The emails of this ContributorIdentity.
        """
        return self._emails

    @emails.setter
    def emails(self, emails: Optional[List[str]]):
        """Sets the emails of this ContributorIdentity.

        Email addresses belonging to the person.

        :param emails: The emails of this ContributorIdentity.
        """
        self._emails = emails

    @property
    def names(self) -> Optional[List[str]]:
        """Gets the names of this ContributorIdentity.

        The person is known as each of these full names. The format is arbitrary.

        :return: The names of this ContributorIdentity.
        """
        return self._names

    @names.setter
    def names(self, names: Optional[List[str]]):
        """Sets the names of this ContributorIdentity.

        The person is known as each of these full names. The format is arbitrary.

        :param names: The names of this ContributorIdentity.
        """
        self._names = names
