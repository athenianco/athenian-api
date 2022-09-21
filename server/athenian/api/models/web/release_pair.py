from athenian.api.models.web.base_model_ import Model


class ReleasePair(Model):
    """A pair of release names within the same repository."""

    old: str
    new: str
