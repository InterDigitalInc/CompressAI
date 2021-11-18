import warnings

warnings.warn(
    "priors module is deprecated, it is renamed 'google'",
    DeprecationWarning,
    stacklevel=2,
)

from .google import *  # noqa: F401, E402
