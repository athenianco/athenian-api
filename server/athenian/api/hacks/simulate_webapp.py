import argparse
import asyncio
from datetime import datetime, timedelta
import time

import aiohttp

from athenian.api.async_utils import gather


def extract_account_id(user):
    """Extract the account id from the user information."""
    admin_accounts = [
        acc_id for acc_id, is_admin in user["accounts"].items() if is_admin
    ]
    non_admin_accounts = [
        acc_id for acc_id, is_admin in user["accounts"].items() if not is_admin
    ]

    return admin_accounts[0] if admin_accounts else non_admin_accounts[0]


async def do_request(client, url, headers, params=None):
    """Run request to `url` with the provided `params` and `headers`."""

    def requester(url):
        if params:
            return client.post(url, headers=headers, json=params)
        else:
            return client.get(url, headers=headers)

    start = datetime.utcnow()
    async with requester(url) as resp:
        end = datetime.utcnow()
        assert resp.status == 200, f"{url} - {await resp.text()}"
        out = await resp.json()

        timing = (end - start).total_seconds()
        headers = resp.headers

        perf_headers = {
            k: v for k, v in headers.items() if k.startswith("X-Performance")
        }

        return (
            out,
            {
                "url": url,
                "headers": perf_headers,
                "timing": timing,
                "start": start,
                "end": end,
                "params": params,
            },
        )


async def get_user(client, headers):
    """Get the user information."""
    url = "https://api.athenian.co/v1/user"
    return await do_request(client, url, headers)


async def get_feature_flags(client, headers, account_id):
    """Get the feature flags associated with the provided account id."""
    url = f"https://api.athenian.co/v1/account/{account_id}/features"
    return await do_request(client, url, headers)


async def get_reposets(client, headers, account_id):
    """Get the reposets associated with the provided account id."""
    url = f"https://api.athenian.co/v1/reposets/{account_id}"
    return await do_request(client, url, headers)


async def get_reposet(client, headers, reposet_id):
    """Get the reposet by id."""
    url = f"https://api.athenian.co/v1/reposet/{reposet_id}"
    return await do_request(client, url, headers)


async def filter_repositories(
    client, headers, account_id, date_from, date_to, exclude_inactive, repos, timezone,
):
    """Filter repositories according to the provided filter paramaters."""
    url = "https://api.athenian.co/v1/filter/repositories"
    params = {
        "account": int(account_id),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "exclude_inactive": exclude_inactive,
        "in": repos,
        "timezone": timezone,
    }
    return await do_request(client, url, headers, params=params)


async def filter_labels(client, headers, account_id, repos):
    """Filter labels according to the provided filter paramaters."""
    url = "https://api.athenian.co/v1/filter/labels"
    params = {
        "account": int(account_id),
        "repositories": repos,
    }
    return await do_request(client, url, headers, params=params)


async def get_teams(client, headers, account_id):
    """Get the teams associated with the provided account id."""
    url = f"https://api.athenian.co/v1/teams/{account_id}"
    return await do_request(client, url, headers)


async def filter_contributors(
    client, headers, account_id, date_from, date_to, repos, timezone,
):
    """Filter contributors according to the provided filter paramaters."""
    url = "https://api.athenian.co/v1/filter/contributors"
    params = {
        "account": int(account_id),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "as": ["author"],
        "in": repos,
        "timezone": timezone,
    }
    return await do_request(client, url, headers, params=params)


async def chained_filter_repositories_labels(
    client, headers, account_id, date_from, date_to, exclude_inactive, repos, timezone,
):
    """Chain the request to get filtered repos and labels."""
    filtered_repositories, fr_datapoint = await filter_repositories(
        client,
        headers,
        int(account_id),
        date_from,
        date_to,
        exclude_inactive,
        repos,
        timezone,
    )
    filtered_labels, fl_datapoint = await filter_labels(
        client, headers, account_id, filtered_repositories,
    )

    return [(filtered_repositories, fr_datapoint), (filtered_labels, fl_datapoint)]


async def chained_get_teams_filter_contributors(
    client, headers, account_id, date_from, date_to, repos, timezone,
):
    """Chain the request to get teams and filtered contributors."""
    teams, teams_datapoint = await get_teams(client, headers, account_id)
    filtered_contributors, fc_datapoint = await filter_contributors(
        client, headers, account_id, date_from, date_to, repos, timezone,
    )

    return [(teams, teams_datapoint), (filtered_contributors, fc_datapoint)]


async def filter_pull_requests(
    client,
    headers,
    account_id,
    date_from,
    date_to,
    exclude_inactive,
    repos,
    contributors,
    labels,
    limit,
    timezone,
):
    """Filter pull requests according to the provided filter paramaters."""
    url = "https://api.athenian.co/v1/filter/pull_requests"
    params = {
        "account": int(account_id),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "exclude_inactive": exclude_inactive,
        "in": repos,
        "labels_include": labels,
        "limit": limit,
        "properties": [
            "wip",
            "reviewing",
            "merging",
            "releasing",
            "release_happened",
            "rejection_happened",
            "force_push_dropped",
        ],
        "timezone": timezone,
        "with": {"author": contributors},
    }
    return await do_request(client, url, headers, params=params)


async def get_histograms_prs(
    client,
    headers,
    account_id,
    date_from,
    date_to,
    exclude_inactive,
    repos,
    contributors,
    labels,
    timezone,
):
    """Get the historam metrics for the PRs filetered with the provided params."""
    url = "https://api.athenian.co/v1/histograms/pull_requests"
    forset = [
        {
            "labels_include": labels,
            "repositories": repos,
            "with": {"author": contributors},
        },
    ]
    histograms = [
        {"metric": m, "scale": "log", "bins": 15}
        for m in (
            "pr-lead-time",
            "pr-wip-time",
            "pr-review-time",
            "pr-merging-time",
            "pr-release-time",
            "pr-size",
        )
    ]
    params = {
        "account": int(account_id),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "exclude_inactive": exclude_inactive,
        "for": forset,
        "histograms": histograms,
        "quantiles": [0.05, 1],
        "timezone": timezone,
    }
    return await do_request(client, url, headers, params=params)


async def get_metrics_prs(
    client,
    headers,
    account_id,
    date_from,
    date_to,
    exclude_inactive,
    granularities,
    repos,
    repogroups,
    contributors,
    labels,
    timezone,
):
    """Get the linear metrics for the PRs filetered with the provided params."""
    url = "https://api.athenian.co/v1/metrics/pull_requests"
    forset = [
        {
            "labels_include": labels,
            "repositories": repos,
            "repogroups": repogroups,
            "with": {"author": contributors},
        },
    ]

    if not repogroups:
        forset[0].pop("repogroups")

    params = {
        "account": int(account_id),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "exclude_inactive": exclude_inactive,
        "for": forset,
        "metrics": [
            "pr-wip-time",
            "pr-wip-count",
            "pr-wip-count-q",
            "pr-review-time",
            "pr-review-count",
            "pr-review-count-q",
            "pr-merging-time",
            "pr-merging-count",
            "pr-merging-count-q",
            "pr-release-time",
            "pr-release-count",
            "pr-release-count-q",
            "pr-lead-time",
            "pr-lead-count",
            "pr-lead-count-q",
            "pr-cycle-time",
            "pr-cycle-count",
            "pr-cycle-count-q",
            "pr-all-count",
            "pr-wait-first-review-time",
            "pr-wait-first-review-count",
            "pr-wait-first-review-count-q",
            "pr-flow-ratio",
            "pr-opened",
            "pr-reviewed",
            "pr-not-reviewed",
            "pr-merged",
            "pr-rejected",
            "pr-closed",
            "pr-done",
            "pr-size",
            "pr-wip-pending-count",
            "pr-review-pending-count",
            "pr-merging-pending-count",
            "pr-release-pending-count",
        ],
        "granularities": granularities,
        "quantiles": [0, 0.95],
        "timezone": timezone,
    }
    return await do_request(client, url, headers, params=params)


async def get_metrics_developers(
    client, headers, account_id, date_from, date_to, repos, contributors, timezone,
):
    """Get the developers' metrics for the PRs filetered with the provided params."""
    url = "https://api.athenian.co/v1/metrics/developers"
    forset = [{"developers": contributors, "repositories": repos}]

    params = {
        "account": int(account_id),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "for": forset,
        "metrics": [
            "dev-commits-pushed",
            "dev-lines-changed",
            "dev-prs-created",
            "dev-prs-reviewed",
            "dev-prs-merged",
            "dev-releases",
            "dev-reviews",
            "dev-review-approvals",
            "dev-review-rejections",
            "dev-review-neutrals",
            "dev-pr-comments",
            "dev-regular-pr-comments",
            "dev-review-pr-comments",
        ],
        "timezone": timezone,
    }
    return await do_request(client, url, headers, params=params)


async def get_metrics_releases(
    client,
    headers,
    account_id,
    date_from,
    date_to,
    granularities,
    repos,
    contributors,
    timezone,
):
    """Get the releases' metrics filetered with the provided params."""
    url = "https://api.athenian.co/v1/metrics/releases"
    forset = [repos]

    params = {
        "account": int(account_id),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "for": forset,
        "metrics": ["release-count", "release-avg-prs", "release-prs"],
        "granularities": granularities,
        "timezone": timezone,
    }
    return await do_request(client, url, headers, params=params)


async def main(
    token, date_from, date_to, exclude_inactive, limit, timezone, cache_enabled,
):
    """Run the benchmarker emulating the refresh of the webapp Overview section."""
    print("=========================")
    print(
        f"Starting benchmark: [{date_from}, {date_to}], exclude_inactive={exclude_inactive}, "
        f"limit={limit}, timezone={timezone}, cache_enabled={cache_enabled}",
    )
    print("=========================")

    headers = {"authorization": f"Bearer {token}"}
    if not cache_enabled:
        headers["cache-control"] = "no-cache"
        headers["pragma"] = "no-cache"

    start = datetime.utcnow()
    async with aiohttp.ClientSession() as client:
        data = []
        user, datapoint = await get_user(client, headers)
        data.append(datapoint)

        account_id = extract_account_id(user)

        feature_flags_out, reposets_out = await gather(
            get_feature_flags(client, headers, account_id),
            get_reposets(client, headers, account_id),
        )
        feature_flags, datapoint = feature_flags_out
        data.append(datapoint)

        reposets, datapoint = reposets_out
        data.append(datapoint)

        all_reposets_out = await gather(
            *[get_reposet(client, headers, reposet["id"]) for reposet in reposets],
        )

        main_reposet = None
        for reposet_out in all_reposets_out:
            reposet, datapoint = reposet_out
            data.append(datapoint)

            if reposet["name"] == "all":
                main_reposet = reposet

        if not main_reposet:
            exit(1)

        chained_filter_repositories_labels_out = await chained_filter_repositories_labels(
            client,
            headers,
            account_id,
            date_from,
            date_to,
            exclude_inactive,
            main_reposet["items"],
            timezone,
        )

        filter_repos_out, filter_labels_out = chained_filter_repositories_labels_out
        filtered_repositories, datapoint = filter_repos_out
        data.append(datapoint)

        labels, datapoint = filter_labels_out
        data.append(datapoint)

        chained_get_teams_filter_contributors_out = await chained_get_teams_filter_contributors(
            client,
            headers,
            account_id,
            date_from,
            date_to,
            main_reposet["items"],
            timezone,
        )

        teams_out, filter_contributors_out = chained_get_teams_filter_contributors_out
        teams, datapoint = teams_out
        data.append(datapoint)

        filtered_contributors, datapoint = filter_contributors_out
        data.append(datapoint)

        date_from_2x = date_from - timedelta(days=1 + (date_to - date_from).days)
        interval_length = (date_to - date_from).days

        if interval_length <= 5 * 7:
            custom_granularity = "day"
        elif interval_length <= 5 * 30:
            custom_granularity = "week"
        else:
            custom_granularity = "month"

        # main parallel calls
        parallel_block_outs = await gather(
            filter_pull_requests(
                client,
                headers,
                account_id,
                date_from,
                date_to,
                exclude_inactive,
                filtered_repositories,
                [fc["login"] for fc in filtered_contributors],
                [],
                limit,
                timezone,
            ),
            get_histograms_prs(
                client,
                headers,
                account_id,
                date_from,
                date_to,
                exclude_inactive,
                filtered_repositories,
                [fc["login"] for fc in filtered_contributors],
                [],
                timezone,
            ),
            get_metrics_prs(
                client,
                headers,
                account_id,
                date_from,
                date_to,
                exclude_inactive,
                ["all"],
                filtered_repositories,
                [[i] for i in range(len(filtered_repositories))],
                [fc["login"] for fc in filtered_contributors],
                [],
                timezone,
            ),
            get_metrics_prs(
                client,
                headers,
                account_id,
                date_from_2x,
                date_to,
                exclude_inactive,
                [f"{interval_length} day", custom_granularity],
                filtered_repositories,
                [],
                [fc["login"] for fc in filtered_contributors],
                [],
                timezone,
            ),
            get_metrics_developers(
                client,
                headers,
                account_id,
                date_from,
                date_to,
                filtered_repositories,
                [fc["login"] for fc in filtered_contributors],
                timezone,
            ),
            get_metrics_releases(
                client,
                headers,
                account_id,
                date_from_2x,
                date_to,
                [f"{interval_length} day", custom_granularity],
                filtered_repositories,
                [fc["login"] for fc in filtered_contributors],
                timezone,
            ),
            filter_contributors(
                client,
                headers,
                account_id,
                date_from_2x,
                date_from - timedelta(days=1),
                filtered_repositories,
                timezone,
            ),
        )

        for o in parallel_block_outs:
            _, datapoint = o
            data.append(datapoint)

    end = datetime.utcnow()
    timing = (end - start).total_seconds()
    print("=========================")
    for d in data:
        print("-------------------------")
        print(f"{d['url']} - {d['timing']}s")
        print("\n".join(f"{k}: {v}" for k, v in d["headers"].items()))
    print("=========================")
    print(f"[{date_from} - {date_to}] Timing: {timing}s ({start} - {end})")
    print("=========================")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, dest="token", help="Bearer token")
    parser.add_argument(
        "--date-from",
        required=False,
        dest="date_from",
        help="Start date of the interval (format: '%%Y-%%m-%%d')",
    )
    parser.add_argument(
        "--date-to",
        required=False,
        dest="date_to",
        help="End date of the interval (format: '%%Y-%%m-%%d')",
    )
    parser.add_argument(
        "--days-timedelta",
        required=False,
        type=int,
        dest="days_timedelta",
        help="Set the interval to [today - timedelta(days=days_timedelta), today]",
    )
    parser.add_argument(
        "--include-inactive",
        required=False,
        action="store_true",
        dest="include_inactive",
        help="Whether to include inactive PRs",
    )
    parser.add_argument(
        "--limit",
        required=False,
        type=int,
        dest="limit",
        default=500,
        help="Limit when calling `/filter/pull_requests`",
    )
    parser.add_argument(
        "--timezone-offset",
        required=False,
        type=int,
        dest="timezone",
        help="Timezone offset in minutes",
    )
    parser.add_argument(
        "--cache-enabled",
        required=False,
        action="store_true",
        dest="cache_enabled",
        help="Whether to enable cache",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.date_from and args.date_to:
        date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
        date_to = datetime.strptime(args.date_to, "%Y-%m-%d")
    elif args.days_timedelta:
        date_to = datetime.utcnow().date()
        date_from = date_to - timedelta(days=args.days_timedelta)
    else:
        exit(1)

    if args.timezone:
        timezone = args.timezone
    else:
        is_dst = time.daylight and time.localtime().tm_isdst > 0
        utc_offset = -(time.altzone if is_dst else time.timezone)
        timezone = int(utc_offset / 60)

    asyncio.run(
        main(
            args.token,
            date_from,
            date_to,
            not args.include_inactive,
            args.limit,
            timezone,
            args.cache_enabled,
        ),
    )
