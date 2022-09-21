from datetime import datetime
from typing import List

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.code_check_run_statistics import CodeCheckRunStatistics


class FilteredCodeCheckRun(Model):
    """Mined information about a code check run."""

    title: str
    repository: str
    last_execution_time: datetime
    last_execution_url: str
    total_stats: CodeCheckRunStatistics
    prs_stats: CodeCheckRunStatistics
    size_groups: List[int]
