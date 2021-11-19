from datetime import date, datetime, timedelta, timezone
import pickle
from typing import Collection, List, Optional, Sequence, Tuple
import warnings

import aiomcache
from dateutil.rrule import MONTHLY, rrule
import numpy as np

from athenian.api.cache import cached, short_term_exptime
from athenian.api.controllers.features.github.check_run_metrics import \
    calculate_check_run_outcome_masks
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.check_run import check_suite_completed_column, \
    mine_check_runs
from athenian.api.controllers.miners.types import CodeCheckRunListItem, CodeCheckRunListStats
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import CheckRun
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, pushers, labels, jira, quantiles, **_:
    (
        time_from.timestamp(), time_to.timestamp(),
        ",".join(sorted(repositories)),
        ",".join(sorted(pushers)),
        labels,
        jira,
        "%s,%s" % tuple(quantiles),
    ),
)
async def filter_check_runs(time_from: datetime,
                            time_to: datetime,
                            repositories: Collection[str],
                            pushers: Collection[str],
                            labels: LabelFilter,
                            jira: JIRAFilter,
                            quantiles: Sequence[float],
                            meta_ids: Tuple[int, ...],
                            mdb: DatabaseLike,
                            cache: Optional[aiomcache.Client],
                            ) -> Tuple[List[date], List[CodeCheckRunListItem]]:
    """
    Gather information about code check runs according to the filters.

    :param time_from: Check runs must launch beginning from this time.
    :param time_to: Check runs must launch ending with this time.
    :param repositories: Check runs must belong to these repositories.
    :param pushers: Check runs must be triggered by these developers.
    :param labels: PR -> label filters. This effectively makes "total" and "prs" stats the same.
    :param jira: PR -> JIRA filters. This effectively makes "total" and "prs" stats the same.
    :param quantiles: Quantiles apply to the execution time distribution before calculating \
                      the means.
    :param meta_ids: Metadata account IDs.
    :param mdb: Metadata DB instance.
    :param cache: Optional memcached client.
    :return: 1. timeline - the X axis of all the charts. \
             2. list of the mined check run type's information and statistics.
    """
    df_check_runs = await mine_check_runs(
        time_from, time_to, repositories, pushers, labels, jira, False,
        meta_ids, mdb, cache)
    timeline = _build_timeline(time_from, time_to)
    timeline_dates = [d.date() for d in timeline.tolist()]
    if df_check_runs.empty:
        return timeline_dates, []
    suite_statuses = df_check_runs[CheckRun.check_suite_status.name].values.astype("S")
    completed = np.nonzero(np.in1d(suite_statuses, [b"COMPLETED", b"SUCCESS", b"FAILURE"]))[0]
    df_check_runs = df_check_runs.take(completed)
    del suite_statuses, completed
    df_check_runs.sort_values(CheckRun.started_at.name, inplace=True, ascending=False)
    repocol = df_check_runs[CheckRun.repository_full_name.name].values.astype("S")
    crnamecol = np.char.encode(df_check_runs[CheckRun.name.name].values.astype("U"), "UTF-8")
    group_keys = np.char.add(np.char.add(repocol, b"|"), crnamecol)
    unique_repo_crnames, first_encounters, inverse_cr_map, repo_crnames_counts = np.unique(
        group_keys, return_counts=True, return_index=True, return_inverse=True)
    unique_repo_crnames = np.char.decode(np.char.partition(unique_repo_crnames, b"|"), "UTF-8")
    started_ats = df_check_runs[CheckRun.started_at.name].values
    last_execution_times = started_ats[first_encounters].astype("datetime64[s]")
    last_execution_urls = df_check_runs[CheckRun.url.name].values[first_encounters]

    suitecol = df_check_runs[CheckRun.check_suite_node_id.name].values
    unique_suites, run_counts = np.unique(suitecol, return_counts=True)
    suite_blocks = np.array(np.split(np.argsort(suitecol), np.cumsum(run_counts)[:-1]),
                            dtype=object)
    unique_run_counts, back_indexes, group_counts = np.unique(
        run_counts, return_inverse=True, return_counts=True)
    run_counts_order = np.argsort(back_indexes)
    ordered_indexes = np.concatenate(suite_blocks[run_counts_order]).astype(int, copy=False)
    suite_size_map = np.zeros(len(df_check_runs), dtype=int)
    suite_size_map[ordered_indexes] = np.repeat(
        unique_run_counts, group_counts * unique_run_counts)

    no_pr_mask = df_check_runs[CheckRun.pull_request_node_id.name].values == 0
    prs_inverse_cr_map = inverse_cr_map.copy()
    prs_inverse_cr_map[no_pr_mask] = -1

    statuscol = df_check_runs[CheckRun.status.name].values.astype("S")
    conclusioncol = df_check_runs[CheckRun.conclusion.name].values.astype("S")
    check_suite_conclusions = \
        df_check_runs[CheckRun.check_suite_conclusion.name].values.astype("S")
    success_mask, failure_mask, skipped_mask = calculate_check_run_outcome_masks(
        statuscol, conclusioncol, check_suite_conclusions, True, True, True)
    commitscol = df_check_runs[CheckRun.commit_node_id.name].values

    started_ats = started_ats.astype("datetime64[s]")
    completed_ats = df_check_runs[CheckRun.completed_at.name].values.astype(started_ats.dtype)
    critical_mask = completed_ats == \
        df_check_runs[check_suite_completed_column].values.astype(started_ats.dtype)
    elapseds = completed_ats - started_ats
    elapsed_mask = elapseds == elapseds
    timeline_masks = (timeline[:-1, None] <= started_ats) & (started_ats < timeline[1:, None])
    timeline_elapseds = np.broadcast_to(elapseds[None, :], (len(timeline) - 1, len(elapseds)))
    all_time_range = (timeline[0] <= started_ats) & (started_ats < timeline[-1])
    nat = np.array("NaT", dtype="timedelta64")

    result = []
    # workaround https://github.com/numpy/numpy/issues/19379
    np.seterr(divide="warn")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            warnings.filterwarnings("ignore", "Mean of empty slice")
            warnings.filterwarnings("ignore", "divide by zero encountered in true_divide")
            for i, ((repo, _, name), last_execution_time, last_execution_url) in enumerate(zip(
                    unique_repo_crnames, last_execution_times, last_execution_urls)):
                masks = {"total": inverse_cr_map == i, "prs": prs_inverse_cr_map == i}
                for k, v in masks.items():
                    v_in_range = v & all_time_range
                    masks[k] = (v_in_range & ~skipped_mask, v_in_range & skipped_mask)
                result.append(CodeCheckRunListItem(
                    title=name,
                    repository=repo,
                    last_execution_time=last_execution_time.item().replace(tzinfo=timezone.utc),
                    last_execution_url=last_execution_url,
                    size_groups=np.unique(suite_size_map[masks["total"][0]]).tolist(),
                    **{f"{key}_stats": CodeCheckRunListStats(
                        count=mask.sum(),
                        successes=success_mask[mask].sum(),
                        skips=skips_mask.sum(),
                        critical=critical_mask[mask].any(),
                        flaky_count=len(np.intersect1d(commitscol[success_mask & mask],
                                                       commitscol[failure_mask & mask])),
                        mean_execution_time=_val_or_none(np.mean((
                            tight_ts := elapseds[elapsed_mask & (
                                qmask := _tighten_mask_by_quantiles(elapseds, mask, quantiles)
                            )]))),
                        stddev_execution_time=_val_or_none(np.round(np.std(tight_ts.view(int)))
                                                           .astype(int).view("timedelta64[s]")
                                                           if len(tight_ts) else nat),
                        median_execution_time=_val_or_none(np.median(
                            elapseds[elapsed_mask & mask])),
                        count_timeline=timeline_masks[:, mask].astype(bool).sum(axis=1).tolist(),
                        successes_timeline=timeline_masks[:, success_mask & mask].astype(
                            bool).sum(axis=1).tolist(),
                        mean_execution_time_timeline=np.mean(
                            timeline_elapseds,
                            where=timeline_masks & qmask[None, :] & elapsed_mask[None, :],
                            axis=1,
                        ).tolist(),
                        # np.median does not have `where` as of 2021
                        median_execution_time_timeline=np.nanmedian(
                            np.where(timeline_masks & mask[None, :],
                                     timeline_elapseds,
                                     np.timedelta64("NaT")),
                            axis=-1,
                        ).tolist(),
                    )
                        for key, (mask, skips_mask) in masks.items()
                    },
                ))
    finally:
        np.seterr(divide="raise")
    return timeline_dates, result


def _build_timeline(time_from: datetime, time_to: datetime) -> np.ndarray:
    days = (time_to - time_from).days
    if days < 5 * 7:
        timeline = np.array([(time_from + timedelta(days=i)) for i in range(days + 1)],
                            dtype="datetime64[s]")
    elif days < 5 * 30:
        timeline = np.array([(time_from + timedelta(days=i)) for i in range(0, days + 6, 7)],
                            dtype="datetime64[s]")
        timeline[-1] = time_to
    else:
        timeline = list(rrule(MONTHLY, dtstart=time_from, until=time_to, bymonthday=1))
        if timeline[0] > time_from:
            timeline.insert(0, time_from)
        if timeline[-1] < time_to:
            timeline.append(time_to)
        timeline = np.array(timeline, dtype="datetime64[s]")
    return timeline


def _tighten_mask_by_quantiles(elapseds: np.ndarray,
                               mask: np.ndarray,
                               quantiles: Sequence[float],
                               ) -> np.ndarray:
    if quantiles[0] == 0 and quantiles[1] == 1:
        return mask
    samples = elapseds[mask]
    if len(samples) == 0:
        return mask
    mask = mask.copy()
    qmin, qmax = np.nanquantile(samples, quantiles, interpolation="nearest")
    if qmin != qmin:
        mask[:] = False
        return mask
    qmask = (samples < qmin) | (samples > qmax)
    mask[np.nonzero(mask)[0][qmask]] = False
    return mask


def _val_or_none(val):
    if val == val:
        return val.item()
    return None
