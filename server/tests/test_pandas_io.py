from datetime import datetime, timezone
import json
from lzma import LZMAFile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.check_run import mine_check_runs
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.pandas_io import deserialize_df, json_dumps, serialize_df


async def test_zero_rows(mdb):
    df = await mine_check_runs(
        datetime(2020, 10, 10, tzinfo=timezone.utc),
        datetime(2020, 10, 23, tzinfo=timezone.utc),
        ["src-d/go-git"],
        [],
        LabelFilter.empty(),
        JIRAFilter.empty(),
        LogicalRepositorySettings.empty(),
        (6366825,),
        mdb,
        None,
    )
    assert df.empty
    assert_frame_equal(df, deserialize_df(serialize_df(df)))


async def test_json_torture():
    with LZMAFile(Path(__file__).with_name("torture.json.xz")) as fin:
        obj = json.loads(fin.read().decode())
    json.loads(json_dumps(obj).decode())


async def test_json_smoke():
    obj = {
        "aaa": ["bb", 123, 100, 1.25, None],
        "bbb": {
            "x": True,
            "y": False,
            "áббц": "zz",
        },
    }
    assert obj == json.loads(json_dumps(obj).decode())


async def test_smoke():
    df = pd.DataFrame(
        {
            "a": np.array(["x", "yy", "zzz"], dtype=object),
            "b": np.array([1, 2002, 3000000003], dtype=int),
            "c": np.array([b"aaa", b"bb", b"c"], dtype="S3"),
            "d": [None, "mom", "dad"],
            "e": [101, None, 303],
            "f": [np.array(["x", "yy", "zzz"], dtype=object)] * 3,
            "g": [np.array(["x", "yy", "zzz"], dtype="S3")] * 3,
            "h": [np.array([1, 2002, 3000000003], dtype=int)] * 3,
            "i": [
                {
                    "api": {"helm_chart": "0.0.103"},
                    "web-app": {"helm_chart": "0.0.53"},
                    "push-api": {"helm_chart": "0.0.29"},
                    "precomputer": {"helm_chart": "0.0.1"},
                },
                {
                    "build": 619,
                    "checksum": "ae46ff6e9059cc1f71086fafd81f0d894deb15d4d18169031df4a5204f434bbc",
                    "job-name": "athenian/metadata/olek",
                },
                {
                    "build": 1448,
                    "author": "gkwillie",
                    "checksum": "4ef9840a007a2187b665add635a9d95daaa26a6165288175d891525f0d70cc6e",
                    "job-name": "athenian/infrastructure/production",
                },
            ],
        },
    )
    new_df = deserialize_df(serialize_df(df))
    assert_frame_equal(df, new_df)
