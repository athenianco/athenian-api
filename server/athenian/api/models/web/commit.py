from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.commit_signature import CommitSignature


class Commit(Model):
    """Information about a commit."""

    def __init__(
        self,
        repository: Optional[str] = None,
        hash: Optional[str] = None,
        author: Optional[CommitSignature] = None,
        committer: Optional[CommitSignature] = None,
        message: Optional[str] = None,
        pushed: Optional[datetime] = None,
        size_added: Optional[int] = None,
        size_removed: Optional[int] = None,
        files_changed: Optional[int] = None,
    ):
        """Commit - a model defined in OpenAPI

        :param repository: The repository of this Commit.
        :param hash: The hash of this Commit.
        :param author: The author of this Commit.
        :param committer: The committer of this Commit.
        :param message: The message of this Commit.
        :param pushed: The pushed of this Commit.
        :param size_added: The size_added of this Commit.
        :param size_removed: The size_removed of this Commit.
        :param files_changed: The files_changed of this Commit.
        """
        self.openapi_types = {
            "repository": str,
            "hash": str,
            "author": CommitSignature,
            "committer": CommitSignature,
            "message": str,
            "pushed": datetime,
            "size_added": int,
            "size_removed": int,
            "files_changed": int,
        }

        self.attribute_map = {
            "repository": "repository",
            "hash": "hash",
            "author": "author",
            "committer": "committer",
            "message": "message",
            "pushed": "pushed",
            "size_added": "size_added",
            "size_removed": "size_removed",
            "files_changed": "files_changed",
        }

        self._repository = repository
        self._hash = hash
        self._author = author
        self._committer = committer
        self._message = message
        self._pushed = pushed
        self._size_added = size_added
        self._size_removed = size_removed
        self._files_changed = files_changed

    @property
    def repository(self) -> str:
        """Gets the repository of this Commit.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :return: The repository of this Commit.
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this Commit.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :param repository: The repository of this Commit.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def hash(self) -> str:
        """Gets the hash of this Commit.

        Git commit hash.

        :return: The hash of this Commit.
        """
        return self._hash

    @hash.setter
    def hash(self, hash: str):
        """Sets the hash of this Commit.

        Git commit hash.

        :param hash: The hash of this Commit.
        """
        if hash is None:
            raise ValueError("Invalid value for `hash`, must not be `None`")

        self._hash = hash

    @property
    def author(self) -> CommitSignature:
        """Gets the author of this Commit.

        :return: The author of this Commit.
        """
        return self._author

    @author.setter
    def author(self, author: CommitSignature):
        """Sets the author of this Commit.

        :param author: The author of this Commit.
        """
        if author is None:
            raise ValueError("Invalid value for `author`, must not be `None`")

        self._author = author

    @property
    def committer(self) -> CommitSignature:
        """Gets the committer of this Commit.

        :return: The committer of this Commit.
        """
        return self._committer

    @committer.setter
    def committer(self, committer: CommitSignature):
        """Sets the committer of this Commit.

        :param committer: The committer of this Commit.
        """
        if committer is None:
            raise ValueError("Invalid value for `committer`, must not be `None`")

        self._committer = committer

    @property
    def message(self) -> str:
        """Gets the message of this Commit.

        Commit message.

        :return: The message of this Commit.
        """
        return self._message

    @message.setter
    def message(self, message: str):
        """Sets the message of this Commit.

        Commit message.

        :param message: The message of this Commit.
        """
        if message is None:
            raise ValueError("Invalid value for `message`, must not be `None`")

        self._message = message

    @property
    def pushed(self) -> datetime:
        """Gets the pushed of this Commit.

        When the commit was pushed to GitHub.

        :return: The pushed of this Commit.
        """
        return self._pushed

    @pushed.setter
    def pushed(self, pushed: datetime):
        """Sets the pushed of this Commit.

        When the commit was pushed to GitHub.

        :param pushed: The pushed of this Commit.
        """
        if pushed is None:
            raise ValueError("Invalid value for `pushed`, must not be `None`")

        self._pushed = pushed

    @property
    def size_added(self) -> int:
        """Gets the size_added of this Commit.

        Overall number of lines added.

        :return: The size_added of this Commit.
        """
        return self._size_added

    @size_added.setter
    def size_added(self, size_added: int):
        """Sets the size_added of this Commit.

        Overall number of lines added.

        :param size_added: The size_added of this Commit.
        """
        if size_added is None:
            raise ValueError("Invalid value for `size_added`, must not be `None`")

        self._size_added = size_added

    @property
    def size_removed(self) -> int:
        """Gets the size_removed of this Commit.

        Overall number of lines removed.

        :return: The size_removed of this Commit.
        """
        return self._size_removed

    @size_removed.setter
    def size_removed(self, size_removed: int):
        """Sets the size_removed of this Commit.

        Overall number of lines removed.

        :param size_removed: The size_removed of this Commit.
        """
        if size_removed is None:
            raise ValueError("Invalid value for `size_removed`, must not be `None`")

        self._size_removed = size_removed

    @property
    def files_changed(self) -> int:
        """Gets the files_changed of this Commit.

        Number of files changed in this PR.

        :return: The files_changed of this Commit.
        """
        return self._files_changed

    @files_changed.setter
    def files_changed(self, files_changed: int):
        """Sets the files_changed of this Commit.

        Number of files changed in this PR.

        :param files_changed: The files_changed of this Commit.
        """
        if files_changed is None:
            raise ValueError("Invalid value for `files_changed`, must not be `None`")

        self._files_changed = files_changed
