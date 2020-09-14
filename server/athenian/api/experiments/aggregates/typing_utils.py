from typing import Dict, Set

PullRequestsCollection = Dict[str, Set[int]]  # repository -> PRs
RepositoryCollection = Dict[int, Set[str]]  # account -> PRs
