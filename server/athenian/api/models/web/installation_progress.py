from datetime import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.table_fetching_progress import TableFetchingProgress
from athenian.api.typing_utils import VerbatimOptional


class InstallationProgress(Model):
    """Data fetching progress of the Athenian metadata retrieval app."""

    started_date: datetime
    finished_date: VerbatimOptional[datetime]
    owner: Optional[str]
    repositories: Optional[int]
    tables: List[TableFetchingProgress]
