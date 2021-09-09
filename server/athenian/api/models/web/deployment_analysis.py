from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_analysis_code import DeploymentAnalysisCode
from athenian.api.models.web.release_set import ReleaseSet


class DeploymentAnalysisUnsealed(Model):
    """Result the analysis of the deployment."""

    openapi_types = {"code": DeploymentAnalysisCode, "releases": ReleaseSet}

    attribute_map = {"code": "code", "releases": "releases"}

    __enable_slots__ = False

    def __init__(self,
                 code: Optional[DeploymentAnalysisCode] = None,
                 releases: Optional[ReleaseSet] = None):
        """DeploymentAnalysis - a model defined in OpenAPI

        :param code: The code of this DeploymentAnalysis.
        :param releases: The releases of this DeploymentAnalysis.
        """
        self._code = code
        self._releases = releases

    @property
    def code(self) -> DeploymentAnalysisCode:
        """Gets the code of this DeploymentAnalysis.

        :return: The code of this DeploymentAnalysis.
        """
        return self._code

    @code.setter
    def code(self, code: DeploymentAnalysisCode):
        """Sets the code of this DeploymentAnalysis.

        :param code: The code of this DeploymentAnalysis.
        """
        if code is None:
            raise ValueError("Invalid value for `code`, must not be `None`")

        self._code = code

    @property
    def releases(self) -> ReleaseSet:
        """Gets the releases of this DeploymentAnalysis.

        :return: The releases of this DeploymentAnalysis.
        """
        return self._releases

    @releases.setter
    def releases(self, releases: ReleaseSet):
        """Sets the releases of this DeploymentAnalysis.

        :param releases: The releases of this DeploymentAnalysis.
        """
        if releases is None:
            raise ValueError("Invalid value for `releases`, must not be `None`")

        self._releases = releases


class DeploymentAnalysis(DeploymentAnalysisUnsealed):
    """Result the analysis of the deployment."""

    __enable_slots__ = True
