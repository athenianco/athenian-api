from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.goal_template import GoalTemplateCommon


class _GoalTemplateCreateRequest(Model, sealed=False):
    """Goal Template creation request."""

    account: int


GoalTemplateCreateRequest = AllOf(
    GoalTemplateCommon,
    _GoalTemplateCreateRequest,
    name="GoalTemplateCreateRequest",
    module=__name__,
)
