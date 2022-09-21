from athenian.api.models.web.base_model_ import Model


class JIRAPriority(Model):
    """JIRA issue priority details."""

    name: str
    image: str
    rank: int
    color: str

    def __lt__(self, other: "JIRAPriority") -> bool:
        """Support sorting."""
        return (self.rank, self.name) < (other.rank, other.name)

    def __hash__(self) -> int:
        """Support dict-s."""
        return hash(self.name)

    def validate_rank(self, rank: int) -> int:
        """Sets the rank of this JIRAPriority.

        Measure of importance (smaller is more important).

        :param rank: The rank of this JIRAPriority.
        """
        if rank is None:
            raise ValueError("Invalid rank for `rank`, must not be `None`")
        if rank < 1:
            raise ValueError(
                "Invalid rank for `rank`, must be a rank greater than or equal to `1`",
            )

        return rank
