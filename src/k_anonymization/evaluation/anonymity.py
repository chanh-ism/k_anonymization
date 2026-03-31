"""
`k`-anonymity privacy metric.
"""

from numpy import ndarray
from pandas import DataFrame


def find_not_k_anonymous_qids(
    data: DataFrame | ndarray, k: int = 2, qids_idx: list = []
):
    """
    Find equivalence classes that violate `k`-anonymity.

    Parameters
    ----------
    data : DataFrame or ndarray
        The data to inspect.
    k : int, default 2
        The privacy parameter `k`.
    qids_idx : list, optional
        The column indices of the QID attributes.
        If not provided, consider all columns as QID attributes.

    Returns
    -------
    list[{qid, count}]
        A list of dictionaries ``{qid, count}``.
    """
    if isinstance(data, ndarray):
        _df = DataFrame(data)
    else:
        _df = data

    if qids_idx == []:
        _qids = _df.keys().to_list()
    else:
        _qids = _df.keys().values[qids_idx].tolist()

    return [
        {"qid": key, "count": value}
        for key, value in _df.groupby(_qids).size().items()
        if value < k
    ]


def get_equivalence_classes(data: DataFrame | ndarray, qids_idx: list = []):
    """
    Get all equivalence classes.

    Parameters
    ----------
    data : DataFrame or ndarray
        The data to inspect.
    qids_idx : list, optional
        The column indices of the QID attributes.
        If not provided, consider all columns as QID attributes.

    Returns
    -------
    list[{qid, count}]
        A list of dictionaries ``{qid, count}``.
    """
    return find_not_k_anonymous_qids(data, float("inf"), qids_idx)


def is_k_anonymous(data: DataFrame | ndarray, k: int = 2, qids_idx: list = []):
    """
    Check whether the data satisfies `k`-anonymity.

    Parameters
    ----------
    data : DataFrame or ndarray
        The data to inspect.
    k : int, default 2
        The privacy parameter `k`.
    qids_idx : list, optional
        The column indices of the QID attributes.
        If not provided, consider all columns as QID attributes.

    Returns
    -------
    bool
        Whether or not the data satisfies `k`-anonymity.
    """
    return get_k_anonymity(data, qids_idx) >= k


def get_k_anonymity(data: DataFrame | ndarray, qids_idx: list = []):
    """
    Get the level of `k`-anonymity of the given data.

    Parameters
    ----------
    data : DataFrame or ndarray
        The data to inspect.
    qids_idx : list, optional
        The column indices of the QID attributes.
        If not provided, consider all columns as QID attributes.

    Returns
    -------
    int
        The privacy parameter `k`.
    """
    if isinstance(data, ndarray):
        _df = DataFrame(data)
    else:
        _df = data

    if qids_idx == []:
        _qids = _df.keys().to_list()
    else:
        _qids = _df.keys().values[qids_idx].tolist()

    return _df.groupby(_qids).size().min()
