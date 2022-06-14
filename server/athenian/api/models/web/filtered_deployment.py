from athenian.api.models.web.base_model_ import AllOf
from athenian.api.models.web.deployment_analysis import DeploymentAnalysisUnsealed
from athenian.api.models.web.deployment_notification import DeploymentNotificationUnsealed

FilteredDeployment = AllOf(
    DeploymentAnalysisUnsealed,
    DeploymentNotificationUnsealed,
    name="FilteredDeployment",
    module=__name__,
)
