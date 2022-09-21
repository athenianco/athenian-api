from datetime import datetime

from athenian.api.models.web.base_model_ import Model


class ListedToken(Model):
    """Details about a token - without the token itself, which is not stored."""

    id: int
    name: str
    last_used: datetime
