from athenian.api.models.web.base_model_ import Enum, Model


class ReleaseMatchStrategy(Model, metaclass=Enum):
    """Release workflow choice: consider certain branch merges or tags as releases."""

    BRANCH = "branch"
    TAG = "tag"
