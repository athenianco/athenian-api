import ctypes
from datetime import timezone
import sys

import pytz

# native extensions load with RTLD_LOCAL by default
# some of our Cython code requires to access invisible symbols
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# Workaround https://github.com/pandas-dev/pandas/issues/32619
pytz.UTC = pytz.utc = timezone.utc
