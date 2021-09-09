from typing import Dict, Optional

from athenian.api.models.web.base_model_ import Model


class DeploymentAnalysisCode(Model):
    """Summary of the deployed code."""

    openapi_types = {
        "prs": Dict[str, int],
        "lines_prs": Dict[str, int],
        "lines_overall": Dict[str, int],
        "commits_prs": Dict[str, int],
        "commits_overall": Dict[str, int],
    }

    attribute_map = {
        "prs": "prs",
        "lines_prs": "lines_prs",
        "lines_overall": "lines_overall",
        "commits_prs": "commits_prs",
        "commits_overall": "commits_overall",
    }

    def __init__(
        self,
        prs: Optional[Dict[str, int]] = None,
        lines_prs: Optional[Dict[str, int]] = None,
        lines_overall: Optional[Dict[str, int]] = None,
        commits_prs: Optional[Dict[str, int]] = None,
        commits_overall: Optional[Dict[str, int]] = None,
    ):
        """DeploymentAnalysisCode - a model defined in OpenAPI

        :param prs: The prs of this DeploymentAnalysisCode.
        :param lines_prs: The lines_prs of this DeploymentAnalysisCode.
        :param lines_overall: The lines_overall of this DeploymentAnalysisCode.
        :param commits_prs: The commits_prs of this DeploymentAnalysisCode.
        :param commits_overall: The commits_overall of this DeploymentAnalysisCode.
        """
        self._prs = prs
        self._lines_prs = lines_prs
        self._lines_overall = lines_overall
        self._commits_prs = commits_prs
        self._commits_overall = commits_overall

    @property
    def prs(self) -> Dict[str, int]:
        """Gets the prs of this DeploymentAnalysisCode.

        Number of deployed pull requests per repository.

        :return: The prs of this DeploymentAnalysisCode.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: Dict[str, int]):
        """Sets the prs of this DeploymentAnalysisCode.

        Number of deployed pull requests per repository.

        :param prs: The prs of this DeploymentAnalysisCode.
        """
        if prs is None:
            raise ValueError("Invalid value for `prs`, must not be `None`")

        self._prs = prs

    @property
    def lines_prs(self) -> Dict[str, int]:
        """Gets the lines_prs of this DeploymentAnalysisCode.

        Number of changed lines in the deployed PRs per repository.

        :return: The lines_prs of this DeploymentAnalysisCode.
        """
        return self._lines_prs

    @lines_prs.setter
    def lines_prs(self, lines_prs: Dict[str, int]):
        """Sets the lines_prs of this DeploymentAnalysisCode.

        Number of changed lines in the deployed PRs per repository.

        :param lines_prs: The lines_prs of this DeploymentAnalysisCode.
        """
        if lines_prs is None:
            raise ValueError("Invalid value for `lines_prs`, must not be `None`")

        self._lines_prs = lines_prs

    @property
    def lines_overall(self) -> Dict[str, int]:
        """Gets the lines_overall of this DeploymentAnalysisCode.

        Number of changed lines in the deployment per repository.

        :return: The lines_overall of this DeploymentAnalysisCode.
        """
        return self._lines_overall

    @lines_overall.setter
    def lines_overall(self, lines_overall: Dict[str, int]):
        """Sets the lines_overall of this DeploymentAnalysisCode.

        Number of changed lines in the deployment per repository.

        :param lines_overall: The lines_overall of this DeploymentAnalysisCode.
        """
        if lines_overall is None:
            raise ValueError("Invalid value for `lines_overall`, must not be `None`")

        self._lines_overall = lines_overall

    @property
    def commits_prs(self) -> Dict[str, int]:
        """Gets the commits_prs of this DeploymentAnalysisCode.

        Number of deployed PR commits per repository.

        :return: The commits_prs of this DeploymentAnalysisCode.
        """
        return self._commits_prs

    @commits_prs.setter
    def commits_prs(self, commits_prs: Dict[str, int]):
        """Sets the commits_prs of this DeploymentAnalysisCode.

        Number of deployed PR commits per repository.

        :param commits_prs: The commits_prs of this DeploymentAnalysisCode.
        """
        if commits_prs is None:
            raise ValueError("Invalid value for `commits_prs`, must not be `None`")

        self._commits_prs = commits_prs

    @property
    def commits_overall(self) -> Dict[str, int]:
        """Gets the commits_overall of this DeploymentAnalysisCode.

        Number of deployed commits per repository.

        :return: The commits_overall of this DeploymentAnalysisCode.
        """
        return self._commits_overall

    @commits_overall.setter
    def commits_overall(self, commits_overall: Dict[str, int]):
        """Sets the commits_overall of this DeploymentAnalysisCode.

        Number of deployed commits per repository.

        :param commits_overall: The commits_overall of this DeploymentAnalysisCode.
        """
        if commits_overall is None:
            raise ValueError("Invalid value for `commits_overall`, must not be `None`")

        self._commits_overall = commits_overall
