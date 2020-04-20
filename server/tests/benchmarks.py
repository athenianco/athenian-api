import argparse
import sys
import time

import pandas as pd
import requests


common_body = {
    "account": 1,
}


queries = [
    ("metrics_2_weeks", "metrics/prs", {
        "for": [{"repositories": ["{1}"]}],
        "metrics": ["pr-lead-time", "pr-cycle-time"],
        "date_from": "2020-04-03",
        "date_to": "2020-04-17",
        "granularities": ["day"],
    }),
    ("metrics_4_weeks", "metrics/prs", {
        "for": [{"repositories": ["{1}"]}],
        "metrics": ["pr-lead-time", "pr-cycle-time"],
        "date_from": "2020-04-03",
        "date_to": "2020-04-17",
        "granularities": ["day"],
    }),
    ("metrics_3_months", "metrics/prs", {
        "for": [{"repositories": ["{1}"]}],
        "metrics": ["pr-lead-time", "pr-cycle-time"],
        "date_from": "2020-01-17",
        "date_to": "2020-04-17",
        "granularities": ["week"],
    }),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="Output CSV path.")
    parser.add_argument("-s", "--server", default="http://0.0.0.0:8080", help="Server address.")
    return parser.parse_args()


def benchmark(addr, endpoint, body):
    payload = common_body.copy()
    payload.update(body)
    url = "%s/v1/%s" % (addr, endpoint)
    start_time = time.time()
    timings = []
    while time.time() - start_time < 60:
        rst = time.time()
        print(requests.post(url, json=payload).text, file=sys.stderr)
        timings.append(time.time() - rst)
        sys.stdout.write(".")
        sys.stdout.flush()
    duration = min(timings)
    print("\n%d calls, min %.3fs" % (len(timings), duration), flush=True)
    return duration


def main():
    args = parse_args()
    results = {}
    for name, endpoint, body in queries:
        print("=== %s ===" % name, flush=True)
        duration = benchmark(args.server, endpoint, body)
        results[name] = duration
    pd.Series(results).to_csv(args.output, header=False)


if __name__ == "__main__":
    exit(main())
