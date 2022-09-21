from datetime import datetime

from athenian.api.models.web.base_model_ import Model


class Share(Model):
    """Saved UI views state with metadata."""

    author: str
    created: datetime
    data: object
