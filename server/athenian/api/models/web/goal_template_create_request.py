from athenian.api.models.web.goal_template import GoalTemplateCommon


class GoalTemplateCreateRequest(GoalTemplateCommon):
    """Goal Template creation request."""

    account: int
