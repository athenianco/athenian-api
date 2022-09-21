from datetime import timedelta
from typing import Union

from athenian.api.models.web.base_model_ import Model


class Interquartile(Model):
    """Middle 50% range."""

    left: Union[float, timedelta]
    right: Union[float, timedelta]
