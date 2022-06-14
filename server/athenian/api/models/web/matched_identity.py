from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.contributor_identity import ContributorIdentity


class MatchedIdentity(Model):
    """Identity mapping of a specific contributor."""

    attribute_types = {
        "from_": ContributorIdentity,
        "to": str,
        "confidence": float,
    }

    attribute_map = {
        "from_": "from",
        "to": "to",
        "confidence": "confidence",
    }

    def __init__(
        self,
        from_: Optional[ContributorIdentity] = None,
        to: Optional[str] = None,
        confidence: Optional[float] = None,
    ):
        """MatchedIdentity - a model defined in OpenAPI

        :param from_: The from_ of this MatchedIdentity.
        :param to: The to of this MatchedIdentity.
        :param confidence: The confidence of this MatchedIdentity.
        """
        self._from_ = from_
        self._to = to
        self._confidence = confidence

    @property
    def from_(self) -> ContributorIdentity:
        """Gets the from_ of this MatchedIdentity.

        :return: The from_ of this MatchedIdentity.
        """
        return self._from_

    @from_.setter
    def from_(self, from_: ContributorIdentity):
        """Sets the from_ of this MatchedIdentity.

        :param from_: The from_ of this MatchedIdentity.
        """
        self._from_ = from_

    @property
    def to(self) -> str:
        """Gets the to of this MatchedIdentity.

        :return: The to of this MatchedIdentity.
        """
        return self._to

    @to.setter
    def to(self, to: str):
        """Sets the to of this MatchedIdentity.

        :param to: The to of this MatchedIdentity.
        """
        self._to = to

    @property
    def confidence(self) -> float:
        """Gets the confidence of this MatchedIdentity.

        :return: The confidence of this MatchedIdentity.
        """
        return self._confidence

    @confidence.setter
    def confidence(self, confidence: float):
        """Sets the confidence of this MatchedIdentity.

        :param confidence: The confidence of this MatchedIdentity.
        """
        self._confidence = confidence
