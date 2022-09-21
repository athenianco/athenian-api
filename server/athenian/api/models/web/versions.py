from athenian.api.models.web.base_model_ import Model


class Versions(Model):
    """Versions of the backend components."""

    api: str
    metadata: str
