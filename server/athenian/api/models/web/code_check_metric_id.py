from athenian.api.models.web.base_model_ import Enum, Model


class CodeCheckMetricID(Model, metaclass=Enum):
    """Currently supported code check metric types."""

    SUITES_COUNT = "chk-suites-count"
    SUITES_IN_PRS_COUNT = "chk-suites-in-prs-count"
    SUCCESSFUL_SUITES_COUNT = "chk-successful-suites-count"
    FAILED_SUITES_COUNT = "chk-failed-suites-count"
    CANCELLED_SUITES_COUNT = "chk-cancelled-suites-count"
    SUCCESS_RATIO = "chk-success-ratio"
    ROBUST_SUITE_TIME = "chk-robust-suite-time"
    SUITE_TIME = "chk-suite-time"
    SUITES_PER_PR = "chk-suites-per-pr"
    SUITE_TIME_PER_PR = "chk-suite-time-per-pr"
    PRS_WITH_CHECKS_COUNT = "chk-prs-with-checks-count"
    FLAKY_COMMIT_CHECKS_COUNT = "chk-flaky-commit-checks-count"
    PRS_MERGED_WITH_FAILED_CHECKS_COUNT = "chk-prs-merged-with-failed-checks-count"
    PRS_MERGED_WITH_FAILED_CHECKS_RATIO = "chk-prs-merged-with-failed-checks-ratio"
    CONCURRENCY = "chk-concurrency"
    CONCURRENCY_MAX = "chk-concurrency-max"
    ELAPSED_TIME_PER_CONCURRENCY = "chk-elapsed-time-per-concurrency"
    SUITE_OCCUPANCY = "chk-suite-occupancy"
    SUITE_CRITICAL_OCCUPANCY = "chk-suite-critical-occupancy"
    SUITE_IMBALANCE = "chk-suite-imbalance"
