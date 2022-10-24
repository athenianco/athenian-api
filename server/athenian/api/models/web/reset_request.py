from typing import Optional

from athenian.api.models.web import Model
from athenian.api.models.web.reset_target import ResetTarget


class ResetRequest(Model):
    """Reset the selected precomputed tables and drop related caches."""

    account: int
    repositories: Optional[list[str]]
    targets: list[str]

    def validate_target(self, targets: list[str]) -> list[str]:
        """Check that each item in `targets` is in `ResetTarget`."""
        if targets is None:
            raise ValueError("Invalid value for `targets`, must not be `None`")
        for target in targets:
            if target not in ResetTarget:
                raise ValueError(f'Target "{target}" must be one of {list(ResetTarget)}')
        return targets
