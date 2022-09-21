from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CommonDeploymentProperties(Model, sealed=False):
    """Define `with_labels` and `without_labels` properties."""

    with_labels: Optional[object]
    without_labels: Optional[object]
