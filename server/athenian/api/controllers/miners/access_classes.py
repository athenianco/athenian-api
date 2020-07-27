from typing import Dict, Type

from athenian.api.controllers.miners.access import AccessChecker
from athenian.api.controllers.miners.github.access import GitHubAccessChecker


# Do not use this to load all the repos for the account! get_account_repositories() instead.
access_classes: Dict[str, Type[AccessChecker]] = {
    "github": GitHubAccessChecker,
}
