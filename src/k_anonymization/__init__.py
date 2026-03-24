"""
Implementation of `k`-anonymization algorithms and evaluation metrics.
"""

from pandas import set_option as pd_set_option

from . import algorithms, core, utils

pd_set_option("mode.copy_on_write", True)

__all__ = ["algorithms", "core", "utils"]
# __all__ += core.__all__
