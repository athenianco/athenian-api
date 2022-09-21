from typing import Optional

from athenian.api.models.web.base_model_ import Model


class LogicalPRRules(Model):
    """Rules to match PRs to logical repository."""

    title: Optional[str]
    labels_include: Optional[list[str]]
