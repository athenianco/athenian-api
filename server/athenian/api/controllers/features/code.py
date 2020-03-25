from dataclasses import dataclass


@dataclass(frozen=True)
class CodeStats:
    """
    Statistics about a certain group of commits: counted "special" commits and LoC which that \
    group contains and the overall counted commits and LoC in that group.

    What's "special" is abstracted here. For example, pushed without making a PR.
    """

    queried_number_of_commits: int
    queried_number_of_lines: int
    total_number_of_commits: int
    total_number_of_lines: int
