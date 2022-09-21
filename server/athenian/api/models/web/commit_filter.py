from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _CommitFilter(Model, sealed=False):
    """Specific parts of the commit filter."""

    in_: (list[str], "in")
    with_author: Optional[list[str]]
    with_committer: Optional[list[str]]
    only_default_branch: Optional[bool]


CommitFilter = AllOf(_CommitFilter, CommonFilterProperties, name="CommitFilter", module=__name__)
