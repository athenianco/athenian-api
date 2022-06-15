import asyncio
import logging

import pytest

from athenian.api.defer import (
    defer,
    enable_defer,
    launch_defer,
    set_defer_loop,
    wait_all_deferred,
    wait_deferred,
)


# GHA blames the termination of AthenianApp, which seems impossible here - we are not using it
@pytest.mark.flaky(reruns=3)
async def test_defer_end_to_end():
    set_defer_loop()
    for _ in range(2):
        done = 0
        done_by_task = {}
        log = logging.getLogger("test_defer")

        async def task(name: str):
            enable_defer(True)
            launch_defer(0.1, "test")
            log.info("task %s launched", name)
            my_done = 0

            async def payload(index: int):
                log.info("test_payload_%s_%d launched", name, index)
                nonlocal done
                done += 1
                await asyncio.sleep(0.05)
                log.info("test_payload_%s_%d woke up", name, index)
                nonlocal my_done
                my_done += 1

            await defer(payload(1), "test_payload_%s_1" % name)
            await defer(payload(2), "test_payload_%s_2" % name)
            await wait_deferred()
            log.info("task %s finished -> %d %d", name, my_done, done)
            done_by_task[name] = my_done

        asyncio.create_task(task("1"))
        asyncio.create_task(task("2"))
        await asyncio.sleep(0.05)
        assert done == 0
        await wait_all_deferred()
        assert done == 4
        while len(done_by_task) < 2:
            await asyncio.sleep(0)
        assert done_by_task == {"1": 2, "2": 2}
