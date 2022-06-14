import pandas as pd

from athenian.api.models.metadata.github import Release

matched_by_column = "matched_by"
index_name = "pull_request_node_id"
release_columns = [
    Release.published_at.name,
    Release.author.name,
    Release.author_node_id.name,
    Release.url.name,
    Release.node_id.name,
    Release.repository_full_name.name,
    matched_by_column,
]


def new_released_prs_df(records=None) -> pd.DataFrame:
    """Create a pandas DataFrame with the required released PR columns."""
    if records is None:
        return pd.DataFrame(
            columns=release_columns,
            index=pd.MultiIndex(
                levels=[pd.Index([]), pd.Index([])],
                codes=[[], []],
                names=[index_name, Release.repository_full_name.name],
            ),
        )
    df = pd.DataFrame.from_records(records, columns=[index_name] + release_columns)
    df.set_index([index_name, Release.repository_full_name.name], inplace=True)
    return df
