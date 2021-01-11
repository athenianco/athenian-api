from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from athenian.api.controllers.miners.types import ReleaseParticipants, ReleaseParticipationKind
from athenian.api.models.metadata.github import PullRequest


def merge_release_participants(participants: List[ReleaseParticipants]) -> ReleaseParticipants:
    """Merge several groups of release participants together."""
    merged = defaultdict(set)
    for dikt in participants:
        for k, v in dikt.items():
            merged[k].update(v)
    return {k: list(v) for k, v in merged.items()}


def group_by_participants(participants: List[ReleaseParticipants],
                          df: pd.DataFrame,
                          ) -> List[np.ndarray]:
    """Triage releases by their contributors."""
    if not participants:
        return [np.arange(len(df))]
    indexes = []
    for group in participants:
        group = group.copy()
        for k, v in group.items():
            group[k] = np.unique(v).astype("U")
        if ReleaseParticipationKind.COMMIT_AUTHOR in group:
            commit_authors = df["commit_authors"].values
            lengths = np.asarray([len(ca) for ca in commit_authors])
            offsets = np.zeros(len(lengths) + 1, dtype=int)
            np.cumsum(lengths, out=offsets[1:])
            commit_authors = np.concatenate(commit_authors).astype("U")
            included_indexes = np.nonzero(np.in1d(
                commit_authors, group[ReleaseParticipationKind.COMMIT_AUTHOR]))[0]
            passed_indexes = np.unique(
                np.searchsorted(offsets, included_indexes, side="right") - 1)
            mask = np.full(len(df), False)
            mask[passed_indexes] = True
            missing_indexes = np.nonzero(~mask)[0]
        else:
            missing_indexes = np.arange(len(df))
        if len(missing_indexes) == 0:
            indexes.append(np.arange(len(df)))
            continue
        if ReleaseParticipationKind.RELEASER in group:
            publishers = df["publisher"].values
            still_missing = np.in1d(
                np.array(publishers[missing_indexes], dtype="U"),
                group[ReleaseParticipationKind.RELEASER],
                invert=True)
            missing_indexes = missing_indexes[still_missing]
        if len(missing_indexes) == 0:
            indexes.append(np.arange(len(df)))
            continue
        if ReleaseParticipationKind.PR_AUTHOR in group:
            key = PullRequest.user_login.key
            pr_authors = [prs[key] for prs in df["prs"].values[missing_indexes]]
            lengths = np.asarray([len(pra) for pra in pr_authors])
            offsets = np.zeros(len(lengths) + 1, dtype=int)
            np.cumsum(lengths, out=offsets[1:])
            pr_authors = np.concatenate(pr_authors).astype("U")
            included_indexes = np.nonzero(np.in1d(
                pr_authors, group[ReleaseParticipationKind.PR_AUTHOR]))[0]
            passed_indexes = np.unique(
                np.searchsorted(offsets, included_indexes, side="right") - 1)
            mask = np.full(len(missing_indexes), False)
            mask[passed_indexes] = True
            missing_indexes = missing_indexes[~mask]
        if len(missing_indexes) == 0:
            indexes.append(np.arange(len(df)))
            continue
        mask = np.full(len(df), True)
        mask[missing_indexes] = False
        indexes.append(np.nonzero(mask)[0])
    return indexes
