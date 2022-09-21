from datetime import timedelta
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class StageTimings(Model):
    """Time spent by the PR in each pipeline stage."""

    wip: timedelta
    review: Optional[timedelta]
    merge: Optional[timedelta]
    release: Optional[timedelta]
    deploy: Optional[dict[str, timedelta]]
