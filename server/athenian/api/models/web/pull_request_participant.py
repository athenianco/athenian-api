from athenian.api.models.web.base_model_ import Model


class PullRequestParticipant(Model):
    """Developer and their role in the PR."""

    STATUS_AUTHOR = "author"
    STATUS_REVIEWER = "reviewer"
    STATUS_COMMIT_AUTHOR = "commit_author"
    STATUS_COMMIT_COMMITTER = "commit_committer"
    STATUS_COMMENTER = "commenter"
    STATUS_MERGER = "merger"
    STATUS_RELEASER = "releaser"
    STATUSES = {
        STATUS_AUTHOR,
        STATUS_REVIEWER,
        STATUS_COMMIT_AUTHOR,
        STATUS_COMMIT_COMMITTER,
        STATUS_COMMENTER,
        STATUS_MERGER,
        STATUS_RELEASER,
    }
    id: str
    status: list[str]

    def __lt__(self, other: "PullRequestParticipant") -> bool:
        """Compute self < other."""
        return self.id < other.id

    def validate_status(self, status: list[str]) -> list[str]:
        """Sets the status of this PullRequestParticipant.

        :param status: The status of this PullRequestParticipant.
        """
        if status is None:
            raise ValueError("Invalid value for `status`, must not be `None`")
        for v in status:
            if v not in self.STATUSES:
                raise ValueError(
                    "Invalid value for `status` (%s), must be one of %s" % (v, self.STATUSES),
                )

        return status
