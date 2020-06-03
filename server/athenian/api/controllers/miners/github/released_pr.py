import pandas as pd

from athenian.api.models.metadata.github import Release


matched_by_column = "matched_by"
index_name = "pull_request_node_id"


def new_released_prs_df(records=None) -> pd.DataFrame:
    """Create a pandas DataFrame with the required released PR columns."""
    columns = [Release.published_at.key,
               Release.author.key,
               Release.url.key,
               Release.repository_full_name.key,
               matched_by_column]
    if records is None:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=index_name))
    return pd.DataFrame.from_records(records, columns=[index_name] + columns, index=index_name)
