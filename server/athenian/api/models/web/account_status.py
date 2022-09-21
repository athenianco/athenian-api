from athenian.api.models.web.base_model_ import Model


class AccountStatus(Model):
    """Status of the user's account membership."""

    is_admin: bool
    expired: bool
    has_ci: bool
    has_jira: bool
    has_deployments: bool
