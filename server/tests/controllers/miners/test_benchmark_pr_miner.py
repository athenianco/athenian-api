import lzma
from pathlib import Path
import pickle

import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.pull_request import PullRequestFactsMiner


async def test_pr_miner_es(benchmark, no_deprecation_warnings, mdb):
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    with lzma.open(Path(__file__).parent / "miner.pickle.xz", "rb") as fin:
        miner = pickle.load(fin)

    def calc():
        for pr in miner:
            facts_miner(pr)

    benchmark(calc)
