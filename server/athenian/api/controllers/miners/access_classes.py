from typing import Dict, Type

from athenian.api.controllers.miners.access import AccessChecker
from athenian.api.controllers.miners.github.access import GitHubAccessChecker


access_classes: Dict[str, Type[AccessChecker]] = {
    "github": GitHubAccessChecker,
}
