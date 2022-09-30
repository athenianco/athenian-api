from athenian.api.models.web.base_model_ import Model


class JIRAStatus(Model):
    """JIRA issue status details."""

    name: str
    stage: str
    project: str

    def __lt__(self, other: "JIRAStatus") -> bool:
        """Support sorting."""
        return self.name < other.name

    def __hash__(self) -> int:
        """Support dict/set keys."""
        return hash((self.name, self.project))

    def validate_stage(self, stage: str) -> str:
        """Sets the stage of this JIRAStatus.

        One of the three status categories.

        :param stage: The stage of this JIRAStatus.
        """
        allowed_values = ("To Do", "In Progress", "Done", "No Category")
        if stage not in allowed_values:
            raise ValueError(
                f"Invalid value for `stage` ({stage}), must be one of {allowed_values}",
            )

        return stage
