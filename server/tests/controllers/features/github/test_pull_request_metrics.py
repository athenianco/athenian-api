from datetime import datetime, timedelta
import itertools

import numpy as np
import pandas as pd
import pytest

from athenian.api.controllers.features.github.pull_request import calculators
from athenian.api.controllers.miners.github.pull_request import Fallback, PullRequestTimes
import athenian.api.controllers.features.github.pull_request_metrics as _  # noqa
from tests.controllers.features.github.test_pull_request import ensure_dtype, pr_samples  # noqa


def random_dropout(pr, prob):
    fields = sorted(PullRequestTimes.__dataclass_fields__)
    killed = np.random.choice(fields, int(len(fields) * prob), replace=False)
    kwargs = {f: getattr(pr, f) for f in fields}
    for k in killed:
        # "created" must always exist
        if k != "created":
            kwargs[k] = Fallback(None, None)
    return PullRequestTimes(**kwargs)


@pytest.mark.parametrize("cls, dtypes",
                         itertools.product(calculators.values(),
                                           ((datetime, timedelta), (pd.Timestamp, pd.Timedelta))))
def test_pull_request_metrics_stability(pr_samples, cls, dtypes):  # noqa: F811
    calc = cls()
    time_from = datetime.utcnow() - timedelta(days=10000)
    time_to = datetime.utcnow()
    for pr in pr_samples(1000):
        pr = random_dropout(ensure_dtype(pr, dtypes[0]), 0.5)
        r = calc.analyze(pr, time_from, time_to)
        assert (r is None) or ((isinstance(r, dtypes[1])) and r > dtypes[1](0)), str(pr)
