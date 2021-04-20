from typing import Callable, Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api.controllers.features.entries import get_calculator
from athenian.api.models.state.models import AccountFeature, Feature, \
    FeatureComponent
from athenian.api.typing_utils import DatabaseLike


async def get_calculator_for_user(
    service: str,
    account_id: int,
    meta_ids: Tuple[int, ...],
    user_id: str,
    god_id: Optional[str],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    pdb: DatabaseLike,
    rdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
    raise_err: Optional[bool] = False,
    base_module: Optional[str] = "athenian.api.experiments",
) -> Callable:
    """Get the metrics calculator function for the given user."""

    def _get_calculator(variation=None):
        return get_calculator(
            service, account_id, meta_ids, mdb, pdb, rdb, cache,
            variation=variation, raise_err=raise_err, base_module=base_module,
        )

    feature_name_prefix = METRIC_ENTRIES_VARIATIONS_PREFIX[service]
    all_metrics_variations_features = await sdb.fetch_all(
        select([Feature.id, Feature.name, Feature.default_parameters]).where(
            and_(
                Feature.name.like(f"{feature_name_prefix}%"),
                Feature.component == FeatureComponent.server,
                Feature.enabled,
            ),
        ),
    )

    if not all_metrics_variations_features:
        return _get_calculator()

    all_metrics_variations_features = {
        row[Feature.id.key]: {
            "name": row[Feature.name.key][len(feature_name_prefix):],
            "params": row[Feature.default_parameters.key] or {},
        }
        for row in all_metrics_variations_features
    }

    metrics_variation_feature = await sdb.fetch_one(
        select([AccountFeature.feature_id, AccountFeature.parameters]).where(
            and_(
                AccountFeature.account_id == account_id,
                AccountFeature.feature_id.in_(all_metrics_variations_features),
                AccountFeature.enabled,
            ),
        ),
    )

    if metrics_variation_feature is None:
        return _get_calculator()

    selected_metrics_variation = all_metrics_variations_features[
        metrics_variation_feature[0]
    ]
    metrics_variation_params = {
        **selected_metrics_variation["params"],
        **(metrics_variation_feature[1] or {}),
    }

    if metrics_variation_params.get("god_only") and not god_id:
        return _get_calculator()

    variation = selected_metrics_variation["name"]
    return _get_calculator(variation=variation)


METRIC_ENTRIES_VARIATIONS_PREFIX = {"github": "github_features_entries_"}
