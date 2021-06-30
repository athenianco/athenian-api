from athenian.api.models.web.base_model_ import Enum, Model


class DeploymentConclusion(Model, metaclass=Enum):
    """State of the completed deployment. Case-insensitive."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
