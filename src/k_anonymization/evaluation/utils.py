from numpy import ndarray
from pandas import DataFrame


def find_not_k_anonymized_qids(
    data: DataFrame | ndarray, k: int = 2, qids_idx: list = []
):
    return get_more_than_k_equivalence_qids(data, k, qids_idx)


def get_equivalence_qids(data: DataFrame | ndarray, qids_idx: list = []):
    return get_more_than_k_equivalence_qids(data, float("inf"), qids_idx)


def get_more_than_k_equivalence_qids(data: DataFrame | ndarray, k, qids_idx: list = []):
    if isinstance(ndarray):
        _df = DataFrame(data)
        _qids = qids_idx
    else:
        _df = data
        _qids = [_df.keys[i] for i in qids_idx]
    return [
        {"qid": k, "count": v} for k, v in _df.groupby(_qids).size().items() if v < k
    ]


def is_k_anonymized(data: DataFrame | ndarray, k: int = 2, qids_idx: list = []):

    if isinstance(ndarray):
        _df = DataFrame(data)
        _qids = qids_idx
    else:
        _df = data
        _qids = [_df.keys[i] for i in qids_idx]

    return _df.groupby(_qids).size().min() >= k
