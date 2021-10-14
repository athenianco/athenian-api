from athenian.api.models.web.base_model_ import Enum, Model


class PullRequestStage(Model, metaclass=Enum):
    """PR's modelled lifecycle stages."""

    WIP = "wip"
    REVIEWING = "reviewing"
    MERGING = "merging"
    RELEASING = "releasing"
    FORCE_PUSH_DROPPED = "force_push_dropped"
    DONE = "done"
    DEPLOYED = "deployed"
