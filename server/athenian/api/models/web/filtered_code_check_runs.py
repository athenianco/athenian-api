from datetime import date
from typing import List

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filtered_code_check_run import FilteredCodeCheckRun


class FilteredCodeCheckRuns(Model):
    """Response from `/filter/code_checks`, found code check runs ordered by repository name and \
    then by title. Note: we always consider the completed run executions only."""

    timeline: List[date]
    items: List[FilteredCodeCheckRun]
