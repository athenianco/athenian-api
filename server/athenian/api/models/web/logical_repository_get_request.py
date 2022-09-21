from athenian.api.models.web.account import _Account
from athenian.api.models.web.base_model_ import AllOf, Model


class _LogicalRepositoryGetRequest(Model, sealed=False):
    name: str


LogicalRepositoryGetRequest = AllOf(
    _LogicalRepositoryGetRequest, _Account, name="LogicalRepositoryGetRequest", module=__name__,
)
