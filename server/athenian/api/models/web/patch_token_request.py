from athenian.api.models.web.base_model_ import Model


class PatchTokenRequest(Model):
    """Request body of `/token/{id}` PATCH. Allows changing the token name."""

    name: str
