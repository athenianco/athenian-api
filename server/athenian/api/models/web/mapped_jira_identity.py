from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class MappedJIRAIdentity(Model):
    """GitHub user (developer) mapped to a JIRA user."""

    developer_id: VerbatimOptional[str]
    developer_name: VerbatimOptional[str]
    jira_name: str
    confidence: float

    def validate_confidence(self, confidence: float) -> float:
        """Sets the confidence of this MappedJIRAIdentity.

        Value from 0 to 1 indicating how similar are the users.

        :param confidence: The confidence of this MappedJIRAIdentity.
        """
        if confidence is None:
            raise ValueError("Invalid value for `confidence`, must not be `None`")
        if confidence > 1:
            raise ValueError(
                "Invalid value for `confidence`, must be a value less than or equal to `1`",
            )
        if confidence < 0:
            raise ValueError(
                "Invalid value for `confidence`, must be a value greater than or equal to `0`",
            )

        return confidence
