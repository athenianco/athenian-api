"""Module to handle account features."""
from datetime import timezone
import json

import sqlalchemy as sa
from sqlalchemy import select

from athenian.api.async_utils import gather
from athenian.api.db import DatabaseLike
from athenian.api.models.state.models import Account, AccountFeature, Feature, FeatureComponent
from athenian.api.models.web import InvalidRequestError, ProductFeature
from athenian.api.response import ResponseError


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


async def get_account_features(account: int, sdb: DatabaseLike) -> list[ProductFeature]:
    """Fetch the account's product features."""

    async def fetch_features():
        account_features = await sdb.fetch_all(
            select(AccountFeature.feature_id, AccountFeature.parameters).where(
                AccountFeature.account_id == account, AccountFeature.enabled,
            ),
        )
        account_features = {row[0]: row[1] for row in account_features}
        features = await sdb.fetch_all(
            select(Feature.id, Feature.name, Feature.default_parameters).where(
                Feature.id.in_(account_features),
                Feature.component == FeatureComponent.webapp,
                Feature.enabled,
            ),
        )
        features = {row[0]: [row[1], row[2]] for row in features}
        return account_features, features

    async def fetch_expires_at():
        expires_at = await sdb.fetch_val(select([Account.expires_at]).where(Account.id == account))
        if sdb.url.dialect == "sqlite":
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at

    (account_features, features), expires_at = await gather(fetch_features(), fetch_expires_at())
    features[-1] = Account.expires_at.name, expires_at

    for k, v in account_features.items():
        try:
            fk = features[k]
        except KeyError:
            continue
        if v is not None:
            if isinstance(v, dict) != isinstance(fk[1], dict):
                raise ResponseError(
                    InvalidRequestError(
                        pointer=f".{fk[0]}.parameters.parameters",
                        detail=(
                            "`parameters` format mismatch: required type"
                            f' {type(fk[1]).__name__} (example: `{{"parameters":'
                            f" {json.dumps(fk[1])}}}`) but got {type(v).__name__} ="
                            f" `{json.dumps(v)}`"
                        ),
                    ),
                )
            if isinstance(v, dict):
                for pk, pv in v.items():
                    fk[1][pk] = pv
            else:
                fk[1] = v
    return [
        ProductFeature(name=name, parameters=parameters)
        for k, (name, parameters) in sorted(features.items())
    ]
