from athenian.api.models.web.base_model_ import Enum, Model


class ReleaseNotificationStatus(Model, metaclass=Enum):
    """What happened to the notification."""

    ACCEPTED_RESOLVED = "accepted-resolved"
    ACCEPTED_PENDING = "accepted-pending"
    IGNORED_DUPLICATE = "ignored-duplicate"
