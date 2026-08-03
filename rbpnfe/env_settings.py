"""
Thread-count settings for the numerical libraries.

BLAS backends (OpenBLAS, MKL) and numba read their thread counts from the
environment **when they are first loaded**. Setting these variables after numpy
has been imported has no effect — the thread pool already exists. This module
must therefore be imported before numpy, which is why ``rbpnfe/__init__.py``
imports it on its first line, before anything else.

By default rbpnfe pins every numerical library to a single thread. This is the
right setting when parallelism comes from worker *processes* — as in
``NucFreeEnergy.eval_landscape(ncores=...)`` or the batch workers— because letting BLAS multithread on top of that
oversubscribes the cores and runs slower than serial.

Opting out
----------
Set ``RBPNFE_ENV_SETTINGS=0`` before importing rbpnfe to leave the environment
untouched and get whatever threading your BLAS defaults to::

    RBPNFE_ENV_SETTINGS=0 python myscript.py

Choosing a different count
--------------------------
Set ``RBPNFE_NUM_THREADS`` to any positive integer::

    RBPNFE_NUM_THREADS=4 python myscript.py

Or, from Python, before importing rbpnfe::

    from rbpnfe.env_settings import set_num_threads
    set_num_threads(4)

Note that ``set_num_threads`` has no effect once numpy has been imported; it
raises a warning in that case.
"""

from __future__ import annotations

import os
import sys
import warnings

__all__ = [
    "THREAD_ENV_VARS",
    "set_num_threads",
    "apply",
    "active_settings",
    "pinning_is_effective",
]

#: False when numpy was already imported at the time the settings were applied,
#: meaning the BLAS thread pool was already built and the variables had no
#: effect. Query via :func:`pinning_is_effective`.
_EFFECTIVE = True

#: Every environment variable consulted by a backend rbpnfe depends on.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",          # OpenMP, used by most BLAS builds
    "OPENBLAS_NUM_THREADS",     # OpenBLAS
    "MKL_NUM_THREADS",          # Intel MKL
    "VECLIB_MAXIMUM_THREADS",   # Apple Accelerate
    "NUMEXPR_NUM_THREADS",      # numexpr
    "NUMBA_NUM_THREADS",        # numba parallel targets
)


def _numpy_already_loaded() -> bool:
    return "numpy" in sys.modules


def set_num_threads(n: int = 1, *, override: bool = True, warn: bool = True) -> None:
    """
    Pin every numerical backend to ``n`` threads.

    Parameters
    ----------
    n
        Thread count. Must be a positive integer.
    override
        When ``False``, variables already present in the environment are left
        as the user set them. When ``True`` (the default) they are overwritten.
    warn
        Emit a warning when numpy is already loaded and the call therefore
        cannot take effect. Suppressed by the automatic import-time call, which
        records the outcome in :func:`pinning_is_effective` instead.

    Warns
    -----
    RuntimeWarning
        If numpy has already been imported, in which case the BLAS thread pool
        is already built and this call cannot change it.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"Thread count must be a positive integer, got {n!r}")

    global _EFFECTIVE
    if _numpy_already_loaded():
        _EFFECTIVE = False
        if warn:
            warnings.warn(
                "numpy is already imported, so the BLAS thread pool is already "
                "built and set_num_threads() cannot change it. Import "
                "rbpnfe.env_settings (or rbpnfe itself) before numpy.",
                RuntimeWarning,
                stacklevel=2,
            )

    for var in THREAD_ENV_VARS:
        if override or var not in os.environ:
            os.environ[var] = str(n)


def pinning_is_effective() -> bool:
    """
    Whether the thread settings were applied before numpy loaded.

    ``False`` means the variables are set but the BLAS thread pool was already
    built, so the process is *not* actually running single-threaded. Worth
    asserting in benchmarks.
    """
    return _EFFECTIVE


def active_settings() -> dict:
    """The current value of every thread variable, for logging and debugging."""
    return {var: os.environ.get(var) for var in THREAD_ENV_VARS}


def apply() -> None:
    """
    Apply the settings described by the environment.

    Honours ``RBPNFE_ENV_SETTINGS`` (``"0"`` disables) and
    ``RBPNFE_NUM_THREADS`` (thread count, default ``1``). Called automatically
    on import.
    """
    if os.environ.get("RBPNFE_ENV_SETTINGS", "1") == "0":
        return

    raw = os.environ.get("RBPNFE_NUM_THREADS", "1")
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(
            f"RBPNFE_NUM_THREADS must be an integer, got {raw!r}"
        ) from None
    if n < 1:
        raise ValueError(f"RBPNFE_NUM_THREADS must be >= 1, got {n}")

    # quiet: importing numpy before rbpnfe is common and a warning on every
    # import would be noise. pinning_is_effective() reports the outcome.
    set_num_threads(n, warn=False)


apply()
