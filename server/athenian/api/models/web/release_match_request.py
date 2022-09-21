from athenian.api.models.web.account import _Account
from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.release_match_setting import _ReleaseMatchSetting


class _ReleaseMatchRequest(Model):
    repositories: list[str]


ReleaseMatchRequest = AllOf(
    _ReleaseMatchSetting,
    _ReleaseMatchRequest,
    _Account,
    name="ReleaseMatchRequest",
    module=__name__,
)
