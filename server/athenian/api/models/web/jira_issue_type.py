from athenian.api.models.web.base_model_ import Model


class JIRAIssueType(Model):
    """Details about a JIRA issue type."""

    name: str
    count: int
    image: str
    project: str
    is_subtask: bool
    is_epic: bool
    normalized_name: str

    def validate_count(self, count: int) -> int:
        """Sets the count of this JIRAIssueType.

        Number of issues that satisfy the filters and belong to this type.

        :param count: The count of this JIRAIssueType.
        """
        if count is None:
            raise ValueError("Invalid value for `count`, must not be `None`")
        if count < 1:
            raise ValueError(
                "Invalid value for `count`, must be a value greater than or equal to `1`",
            )

        return count
