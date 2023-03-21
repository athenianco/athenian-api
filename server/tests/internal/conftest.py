import csv
import gzip
from pathlib import Path

import medvedi as md
import numpy as np
import pytest

from athenian.api.async_utils import infer_dtype
from athenian.api.internal.miners.github.check_run import check_suite_started_column
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.metadata.github import CheckRun


@pytest.fixture(scope="session")
def logical_settings_labels():
    return LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {
                "labels": ["enhancement", "performance", "plumbing", "ssh", "documentation"],
            },
            "src-d/go-git/beta": {"labels": ["bug", "windows"]},
        },
        {},
    )


@pytest.fixture(scope="module")
def alternative_check_run_facts() -> md.DataFrame:
    dtype, *_ = infer_dtype(CheckRun.__table__.columns)
    dtype = {k: v[0] for k, v in dtype.fields.items()}
    dtype["pull_request_merged"] = np.dtype(bool)
    dtype[check_suite_started_column] = np.dtype("datetime64[us]")

    with gzip.open(
        Path(__file__).parent / "features" / "github" / "check_runs.csv.gz", "rt",
    ) as fin:
        index = fin.readline().rstrip().lstrip(",").split(",")
        columns = {k: [] for k in index}
        for record in csv.reader(fin, dialect="unix"):
            for i, v in zip(index, record[1:]):
                if not i:
                    continue
                i_dtype = dtype[i]
                if not v and i_dtype.kind != "S" and i_dtype.kind != "U":
                    columns[i].append(None)
                    continue
                if v.endswith("+00:00"):
                    v = v[:-6]
                columns[i].append(i_dtype.type(v))
    for k, v in columns.items():
        try:
            columns[k] = np.array(v, dtype=dtype[k])
        except TypeError:
            continue
    return md.DataFrame(columns)
