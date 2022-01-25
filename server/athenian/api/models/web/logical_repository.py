import re
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.logical_deployment_rules import LogicalDeploymentRules
from athenian.api.models.web.logical_pr_rules import LogicalPRRules
from athenian.api.models.web.release_match_setting import ReleaseMatchSetting


class _LogicalRepository(Model):
    __enable_slots__ = False

    openapi_types = {
        "name": str,
        "parent": str,
        "prs": LogicalPRRules,
        "releases": ReleaseMatchSetting,
        "deployments": LogicalDeploymentRules,
    }

    attribute_map = {
        "name": "name",
        "parent": "parent",
        "prs": "prs",
        "releases": "releases",
        "deployments": "deployments",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        parent: Optional[str] = None,
        prs: Optional[LogicalPRRules] = None,
        releases: Optional[ReleaseMatchSetting] = None,
        deployments: Optional[LogicalDeploymentRules] = None,
    ):
        """LogicalRepository - a model defined in OpenAPI

        :param name: The name of this LogicalRepository.
        :param parent: The parent of this LogicalRepository.
        :param prs: The prs of this LogicalRepository.
        :param releases: The releases of this LogicalRepository.
        :param deployments: The deployments of this LogicalRepository.
        """
        self._name = name
        self._parent = parent
        self._prs = prs
        self._releases = releases
        self._deployments = deployments

    @property
    def name(self) -> str:
        """Gets the name of this LogicalRepository.

        The logical part of the repository name. Compared to GitHub repository name requirements,
        we additionally allow / to express the hierarchy further.

        :return: The name of this LogicalRepository.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this LogicalRepository.

        The logical part of the repository name. Compared to GitHub repository name requirements,
        we additionally allow / to express the hierarchy further.

        :param name: The name of this LogicalRepository.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, may not be null")
        if len(name) > 100:
            raise ValueError(
                "Invalid value for `name`, length must be less than or equal to `100`")
        if not re.fullmatch(r"([a-zA-Z0-9-_]+\/?)*[a-zA-Z0-9-_]+", name):
            raise ValueError(
                r"Invalid value for `name`, must be a follow pattern or equal to "
                r"`/^([a-zA-Z0-9-_]+\/?)*[a-zA-Z0-9-_]+$/`")

        self._name = name

    @property
    def parent(self) -> str:
        """Gets the parent of this LogicalRepository.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :return: The parent of this LogicalRepository.
        """
        return self._parent

    @parent.setter
    def parent(self, parent: str):
        """Sets the parent of this LogicalRepository.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :param parent: The parent of this LogicalRepository.
        """
        if parent is None:
            raise ValueError("Invalid value for `parent`, may not be null")
        self._parent = parent

    @property
    def prs(self) -> LogicalPRRules:
        """Gets the prs of this LogicalRepository.

        :return: The prs of this LogicalRepository.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: LogicalPRRules):
        """Sets the prs of this LogicalRepository.

        :param prs: The prs of this LogicalRepository.
        """
        if prs is None:
            raise ValueError("Invalid value for `prs`, may not be null")

        self._prs = prs

    @property
    def releases(self) -> ReleaseMatchSetting:
        """Gets the releases of this LogicalRepository.

        :return: The releases of this LogicalRepository.
        """
        return self._releases

    @releases.setter
    def releases(self, releases: ReleaseMatchSetting):
        """Sets the releases of this LogicalRepository.

        :param releases: The releases of this LogicalRepository.
        """
        if releases is None:
            raise ValueError("Invalid value for `releases`, may not be null")
        self._releases = releases

    @property
    def deployments(self) -> LogicalDeploymentRules:
        """Gets the deployments of this LogicalRepository.

        :return: The deployments of this LogicalRepository.
        """
        return self._deployments

    @deployments.setter
    def deployments(self, deployments: LogicalDeploymentRules):
        """Sets the deployments of this LogicalRepository.

        :param deployments: The deployments of this LogicalRepository.
        """
        self._deployments = deployments


class LogicalRepository(_LogicalRepository):
    """
    Set of rules to match PRs, releases, and deployments that has a name and \
    pretends to be a regular GitHub repository everywhere in UI and API requests.

    Release settings are also visible in `/settings/release_match`.
    """

    __enable_slots__ = True
