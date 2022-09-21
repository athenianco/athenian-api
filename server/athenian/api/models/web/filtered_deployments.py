from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filtered_deployment import FilteredDeployment
from athenian.api.models.web.release_set import ReleaseSetInclude


class FilteredDeployments(Model):
    """Found deployments, response from `/filter/deployments`."""

    include: ReleaseSetInclude
    deployments: list[FilteredDeployment]
