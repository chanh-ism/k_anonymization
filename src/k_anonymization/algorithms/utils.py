"""
Utility functions for anonymization algorithms.
"""

import pandas as pd
from numpy import ndarray

from k_anonymization.core import Hierarchy


def generalize_column(
    values: list | ndarray | pd.Series,
    hierarchy: Hierarchy,
    level_from: int,
    level_to: int,
):
    """
    Generalize a column of data.

    Parameters
    ----------
    values : list, numpy.ndarray, or pandas.Series
        The input data to be generalized.
    hierarchy : Hierarchy
        Hierarchy definition for the input data.
    level_from : int
        The current generalization level of the input data.
    level_to : int
        The target generalization level to transform the input data into.

    Returns
    -------
    tuple
        ``(generalized_values, is_suppressed)``

        generalized_values : list, numpy.ndarray, or pandas.Series
            The generalized data in the same format as the input.
        is_suppressed : bool
            Whether the data has been suppressed (replaced all values with '*').

    Raises
    ------
    AssertionError
        If ``level_from`` is not lower than ``level_to``, or if they
        exceed the height of the hierarchy.
    """
    assert level_from < level_to, "level_from must be lower than level_to"
    assert (
        level_from <= hierarchy.height and level_to <= hierarchy.height
    ), "level_from and level_to must not be higher than hierarchy.height"
    if isinstance(values, list) or isinstance(values, ndarray):
        _values = pd.Series(values, name="org")
    else:
        _values = values.copy()
        _values.name = "org"

    _generalized_values = pd.merge(
        _values,
        hierarchy.hierarchy_df[[level_from, level_to]].drop_duplicates(),
        left_on="org",
        right_on=level_from,
    )[level_to]
    _is_suppressed = _generalized_values[0] == "*"

    if isinstance(values, list):
        return _generalized_values.to_list(), _is_suppressed
    elif isinstance(values, ndarray):
        return _generalized_values.to_numpy(), _is_suppressed
    else:
        return _generalized_values, _is_suppressed
