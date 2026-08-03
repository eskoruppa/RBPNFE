"""
RBPNFE
=====

A python module for the evaluation of nucleosome positioning free energies.

"""

# MUST come before any import that pulls in numpy: BLAS and numba read their
# thread counts from the environment when they are first loaded, so setting
# them later has no effect. Disable with RBPNFE_ENV_SETTINGS=0, change the
# count with RBPNFE_NUM_THREADS=n. See rbpnfe/env_settings.py.
from . import env_settings  # noqa: F401  (import order is deliberate)

from .SO3 import so3
from .hcmodel import hc_free_energy
from .scmodel import sc_free_energy
from .free_energy import NucFreeEnergy
from .multiharmonic import MultiharmonicResult, NucleosomeBreath