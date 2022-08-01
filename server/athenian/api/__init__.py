import ctypes
from datetime import timezone
from distutils.version import Version
import os
import sys

import pytz

# native extensions load with RTLD_LOCAL by default
# some of our Cython code requires to access invisible symbols
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# Workaround https://github.com/pandas-dev/pandas/issues/32619
pytz.UTC = pytz.utc = timezone.utc

is_testing = "pytest" in sys.modules or os.getenv("SENTRY_ENV", "development") in (
    "development",
    "test",
)


def _version_init_without_warnings(self, vstring=None):
    """Curse the author of  https://github.com/pypa/distutils/pull/75"""  # noqa: D400
    if vstring:
        self.parse(vstring)


Version.__init__ = _version_init_without_warnings
del _version_init_without_warnings
