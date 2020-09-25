import logging
import runpy
import sys
import traceback

import sentry_sdk

from athenian.api import setup_context


def main() -> int:
    """Emulate `python -m ...`.  If it raises an exception, forward it to Sentry."""
    setup_context(logging.getLogger())
    original_argv = sys.argv.copy()
    sys.argv.pop(0)
    try:
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    except BaseException as e:
        sentry_sdk.capture_exception(e)
        traceback.print_exc()
        return 1
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    sys.exit(main())
