from typing import List

from athenian.api.models.web.base_model_ import Model


class DeleteEventsCacheRequest(Model):
    """Definition of the cache reset operation."""

    account: int
    repositories: List[str]
    targets: List[str]

    def validate_targets(self, targets: list[str]) -> list[str]:
        """Sets the targets of this DeleteEventsCacheRequest.

        Parts of the precomputed cache to reset.

        :param targets: The targets of this DeleteEventsCacheRequest.
        """
        allowed_values = {"release", "deployment"}
        if not set(targets).issubset(set(allowed_values)):
            raise ValueError(
                "Invalid values for `targets` [%s], must be a subset of [%s]"
                % (", ".join(set(targets) - set(allowed_values)), ", ".join(allowed_values)),
            )

        return targets
