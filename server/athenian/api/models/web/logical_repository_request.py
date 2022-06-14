from athenian.api.models.web.account import _Account
from athenian.api.models.web.base_model_ import AllOf
from athenian.api.models.web.logical_repository import _LogicalRepository

LogicalRepositoryRequest = AllOf(
    _LogicalRepository, _Account, name="LogicalRepositoryRequest", module=__name__
)
