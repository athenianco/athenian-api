import numpy as np
from numpy.testing import assert_array_equal

from athenian.api.controllers.jira_controller.common import resolve_acknowledge_time
from athenian.api.models.metadata.jira import Status


class TestResolveAcknowledgeTime:
    def test_smoke(self) -> None:
        created = np.array([2, 1, 1], dtype="datetime64[s]")
        work_began = np.array([3, None, 3], dtype="datetime64[s]")
        prs_began = np.array([None, None, None], dtype="datetime64[s]")
        statuses = np.array(
            [Status.CATEGORY_IN_PROGRESS, Status.CATEGORY_TODO, Status.CATEGORY_DONE],
            dtype=object,
        )
        now = np.datetime64(5, "s")

        res = resolve_acknowledge_time(created, work_began, prs_began, statuses, now)
        expected = np.array([1, 4, 2], dtype="timedelta64[s]")

        assert_array_equal(res, expected)
