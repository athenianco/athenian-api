from typing import Dict, Iterable, Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.controllers.features.entries import make_calculator, MetricEntriesCalculator
from athenian.api.db import DatabaseLike
from athenian.api.models.state.models import AccountFeature, Feature, FeatureComponent
from athenian.api.tracing import sentry_span


@sentry_span
async def get_calculators_for_account(
    services: Iterable[str],
    account_id: int,
    meta_ids: Tuple[int, ...],
    god_id: Optional[str],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    pdb: DatabaseLike,
    rdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
    instrument: Optional[Dict[str, str]] = None,
    base_module: Optional[str] = "athenian.api.experiments",
) -> Dict[str, MetricEntriesCalculator]:
    """Get the species calculator function for the given account."""
    calcs = await gather(*(_get_calculator_for_account(
        s, account_id, meta_ids, god_id, sdb, mdb, pdb, rdb, cache,
        instrument=instrument, base_module=base_module)
        for s in services))
    return dict(zip(services, calcs))


async def _get_calculator_for_account(
    service: str,
    account_id: int,
    meta_ids: Tuple[int, ...],
    god_id: Optional[str],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    pdb: DatabaseLike,
    rdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
    instrument: Optional[Dict[str, str]] = None,
    base_module: Optional[str] = "athenian.api.experiments",
) -> MetricEntriesCalculator:
    def _make_calculator(variation=None):
        if instrument is not None:
            instrument[service] = variation or "default"

        assert service == "github", "we don't support others"
        return make_calculator(
            account_id, meta_ids, mdb, pdb, rdb, cache,
            variation=variation, base_module=base_module,
        )

    feature_name_prefix = METRIC_ENTRIES_VARIATIONS_PREFIX.get(service)
    if not feature_name_prefix:
        return _make_calculator()

    all_metrics_variations_features = await sdb.fetch_all_safe(
        select([Feature.id, Feature.name, Feature.default_parameters]).where(
            and_(
                Feature.name.like(f"{feature_name_prefix}%"),
                Feature.component == FeatureComponent.server,
                Feature.enabled,
            ),
        ),
    )

    if not all_metrics_variations_features:
        return _make_calculator()

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
        return _make_calculator()

    selected_metrics_variation = all_metrics_variations_features[
        metrics_variation_feature[0]
    ]
    metrics_variation_params = {
        **selected_metrics_variation["params"],
        **(metrics_variation_feature[1] or {}),
    }

    if metrics_variation_params.get("god_only") and not god_id:
        return _make_calculator()

    variation = selected_metrics_variation["name"]
    return _make_calculator(variation=variation)


METRIC_ENTRIES_VARIATIONS_PREFIX = {"github": "github_features_entries_"}
