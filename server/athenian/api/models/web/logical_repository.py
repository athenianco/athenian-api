import re
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.logical_deployment_rules import LogicalDeploymentRules
from athenian.api.models.web.logical_pr_rules import LogicalPRRules
from athenian.api.models.web.release_match_setting import ReleaseMatchSetting


class _LogicalRepository(Model, sealed=False):
    name: str
    parent: str
    prs: LogicalPRRules
    releases: ReleaseMatchSetting
    deployments: Optional[LogicalDeploymentRules]

    def validate_name(self, name: str) -> str:
        """Sets the name of this LogicalRepository.

        The logical part of the repository name. Compared to GitHub repository name requirements,
        we additionally allow / to express the hierarchy further.

        :param name: The name of this LogicalRepository.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, may not be null")
        if len(name) > 100:
            raise ValueError(
                "Invalid value for `name`, length must be less than or equal to `100`",
            )
        if not re.fullmatch(r"([a-zA-Z0-9-_]+\/?)*[a-zA-Z0-9-_]+", name):
            raise ValueError(
                r"Invalid value for `name`, must be a follow pattern or equal to "
                r"`/^([a-zA-Z0-9-_]+\/?)*[a-zA-Z0-9-_]+$/`",
            )

        return name


class LogicalRepository(_LogicalRepository):
    """
    Set of rules to match PRs, releases, and deployments that has a name and \
    pretends to be a regular GitHub repository everywhere in UI and API requests.

    Release settings are also visible in `/settings/release_match`.
    """
