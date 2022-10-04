from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_release import CommonRelease


class _DeployedRelease(Model):
    """Specific information about a deployed repository release."""

    prs: int


DeployedRelease = AllOf(CommonRelease, _DeployedRelease, name="DeployedRelease", module=__name__)
