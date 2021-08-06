import gzip
import logging
import sys

from flogging import flogging
import numpy as np
import pandas as pd

from athenian.api.controllers.features.github.pull_request_metrics import \
    PullRequestMetricCalculatorEnsemble
from athenian.api.controllers.miners.types import PullRequestFacts
from athenian.api.models.web import PullRequestMetricID
from athenian.api.typing_utils import df_from_structs


def main():
    """
    Calculate review and lead times for all the PRs fetched by SQL.

    psql -h 0.0.0.0 -p 5432 --db precomputed -U production-cloud-sql -c 'select acc_id, data from github.done_pull_request_facts' -t -A -F"," | tail -n +2 | gzip >done_facts.csv.gz
    """  # noqa
    if len(sys.argv) != 3:
        print("Usage: script.py input.csv.gz output.pickle", file=sys.stderr)
        return 1
    flogging.setup()
    log = logging.getLogger("time_by_lines")
    accs = []
    node_ids = []
    facts = []
    log.info("Reading %s", sys.argv[1])
    with gzip.open(sys.argv[1]) as fin:
        for line in fin:
            acc, node_id, data = line.strip().split(b",", 2)
            accs.append(int(acc.decode()))
            node_ids.append(node_id)
            facts.append(PullRequestFacts(bytes.fromhex(data[2:].decode())))
    log.info("Assembling the data frame")
    df = df_from_structs(facts)
    log.info("Calculating the timedeltas")
    ensemble = PullRequestMetricCalculatorEnsemble(
        PullRequestMetricID.PR_REVIEW_TIME, PullRequestMetricID.PR_LEAD_TIME,
        quantiles=(0, 1))
    ensemble(df,
             np.array(["1970-01-01"], dtype="datetime64[ns]"),
             np.array(["2020-01-01"], dtype="datetime64[ns]"),
             [np.arange(len(df))])
    log.info("Building the result")
    review_times = ensemble[PullRequestMetricID.PR_REVIEW_TIME].peek.astype("timedelta64[s]")
    lead_times = ensemble[PullRequestMetricID.PR_LEAD_TIME].peek.astype("timedelta64[s]")
    df = pd.DataFrame({
        "acc": accs,
        "size": df["size"],
        "node_id": node_ids,
        "review_time": review_times.squeeze(),
        "lead_time": lead_times.squeeze(),
    })
    log.info("Writing the result")
    df.to_pickle(sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
