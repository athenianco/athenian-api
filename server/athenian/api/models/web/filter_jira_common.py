from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _FilterJIRACommon(Model, sealed=False):
    """Common properies if a JIRA issue or epic."""

    priorities: Optional[List[str]]
    types: Optional[List[str]]
    projects: Optional[List[str]]
    labels_include: Optional[List[str]]
    labels_exclude: Optional[List[str]]
    exclude_inactive: bool


FilterJIRACommon = AllOf(
    _FilterJIRACommon,
    CommonFilterProperties,
    name="FilterJIRACommon",
    module=__name__,
    sealed=False,
)
