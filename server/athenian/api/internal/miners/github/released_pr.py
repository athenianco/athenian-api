from itertools import chain

import medvedi as md
import numpy as np
from numpy import typing as npt

from athenian.api.async_utils import infer_dtype
from athenian.api.models.metadata.github import Release

matched_by_column = "matched_by"
release_index_name = "pull_request_node_id"
_release_columns = [
    Release.published_at,
    Release.author,
    Release.author_node_id,
    Release.url,
    Release.node_id,
    Release.repository_full_name,
    matched_by_column,
]
release_columns = [(c.name if not isinstance(c, str) else c) for c in _release_columns]
release_dtypes = {k: v[0] for k, v in infer_dtype(_release_columns)[0].fields.items()}
release_dtypes[release_index_name] = int
release_dtypes[matched_by_column] = int


def new_released_prs_df(columns: dict[str, npt.ArrayLike] | None = None) -> md.DataFrame:
    """Create a pandas DataFrame with the required released PR columns."""
    if columns is None or len(columns[release_index_name]) == 0:
        columns = {
            k: np.array([], dtype=release_dtypes[k])
            for k in chain(release_columns, [release_index_name])
        }
    df = md.DataFrame(columns, index=(release_index_name, Release.repository_full_name.name))
    return df
