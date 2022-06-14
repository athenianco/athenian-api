from typing import List, Optional


def validate_quantiles(quantiles: Optional[List[float]]) -> None:
    """Ensure that the specified two numbers are correct quantiles."""
    if quantiles is None:
        return
    if len(quantiles) != 2:
        raise ValueError("Invalid value for `quantiles`: the length must be 2")
    if quantiles[0] >= quantiles[1]:
        raise ValueError("`quantiles[1]` must be greater than `quantiles[0]`: %s" % quantiles)
    if quantiles[0] < 0 or quantiles[0] > 1:
        raise ValueError("`quantiles[0]` must lie between 0 and 1")
    if quantiles[1] < 0 or quantiles[1] > 1:
        raise ValueError("`quantiles[1]` must lie between 0 and 1")
