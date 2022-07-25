"""Module to handle account features."""

import sqlalchemy as sa

from athenian.api.db import DatabaseLike
from athenian.api.models.state.models import AccountFeature, Feature, FeatureComponent


async def is_feature_enabled(account: int, name: str, sdb: DatabaseLike) -> bool:
    """Return whether a feature is enabled for the given account."""
    enabled = False
    global_row = await sdb.fetch_one(
        sa.select(Feature.id, Feature.enabled).where(
            Feature.name == name, Feature.component == FeatureComponent.server,
        ),
    )
    if global_row is not None:
        feature_id = global_row[Feature.id.name]
        default_enabled = global_row[Feature.enabled.name]
        enabled = await sdb.fetch_val(
            sa.select(AccountFeature.enabled).where(
                AccountFeature.account_id == account,
                AccountFeature.feature_id == feature_id,
            ),
        )
        enabled = (enabled is None and default_enabled) or bool(enabled)
    return enabled
