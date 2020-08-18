from athenian.api.models.web.base_model_ import Enum, Model


class ReleaseMetricID(Model, metaclass=Enum):
    """Linear release metric identifier."""

    RELEASE_COUNT = "release-count"
    RELEASE_SIZE = "release-size"
    RELEASE_AGE = "release-age"
