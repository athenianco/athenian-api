from athenian.api.models.web.base_model_ import Enum, Model


class HistogramScale(Model, metaclass=Enum):
    """X axis scale."""

    LINEAR = "linear"
    LOG = "log"
