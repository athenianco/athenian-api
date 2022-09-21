from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.mapped_jira_identity_change import MappedJIRAIdentityChange


class SetMappedJIRAIdentitiesRequest(Model):
    """Request body of `/settings/jira/identities`. Describes a patch to the GitHub<>JIRA \
    identity mapping."""

    account: int
    changes: list[MappedJIRAIdentityChange]
