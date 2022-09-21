from athenian.api.models.web.base_model_ import Model


class TableFetchingProgress(Model):
    """Used in InstallationProgress.tables to indicate how much data has been loaded in each DB \
    table."""

    fetched: int
    name: str
    total: int

    def __lt__(self, other: "TableFetchingProgress") -> bool:
        """Implement "<"."""
        return self.name < other.name
