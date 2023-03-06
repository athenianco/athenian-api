from enum import Enum

from athenian.api.models.web.base_model_ import Model


class JIRAStatusCategory(Enum):
    """The category of a JIRA issue status."""

    TO_DO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"
    NO_CATEGORY = "No Category"


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
        if stage not in (allowed := [f.value for f in JIRAStatusCategory]):
            raise ValueError(
                f"Invalid value for `stage` ({stage}), must be one of {list(allowed)}",
            )

        return stage
