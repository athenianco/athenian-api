from athenian.api.models.web.base_model_ import Model


class ReleaseNames(Model):
    """Repository name and a list of release names in that repository."""

    repository: str
    names: list[str]
