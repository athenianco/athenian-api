from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.work_type import WorkType


class WorkTypePutRequest(Model):
    """Request body of `PUT /settings/work_type`."""

    account: int
    work_type: WorkType
