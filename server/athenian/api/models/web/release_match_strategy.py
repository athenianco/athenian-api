from athenian.api.models.web.base_model_ import Enum, Model


class ReleaseMatchStrategy(Model, metaclass=Enum):
    """Release workflow choice: how we should detect releases.

    * branch: merges to certain branches are considered releases and nothing else.
    * tag: tags with certain name patterns are considered releases and nothing else.
    * tag_or_branch: follow "tag"; if the repository does not have tags, fallback to "branch".
    """

    BRANCH = "branch"
    TAG = "tag"
    TAG_OR_BRANCH = "tag_or_branch"
    EVENT = "event"
