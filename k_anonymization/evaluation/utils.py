from numpy import ndarray, unique
from pandas.core.frame import DataFrame


def find_not_k_anonymized_qids(
    data: DataFrame | ndarray, k: int = 2, qids_idx: list = [], get_qids: bool = True
):
    return count_equivalent_qids(data, k, qids_idx, get_qids)


def count_equivalent_qids(
    data: DataFrame | ndarray, k: int = -1, qids_idx: list = [], get_qids: bool = False
):
    if len(qids_idx) == 0:
        qids_idx = range(0, data.shape[1])

    def do_count_equivalent_qids(np_arr, k, qids_idx, current_qid_idx, get_qids):
        if current_qid_idx < len(qids_idx):
            unique_values = unique(np_arr.T[qids_idx[current_qid_idx]])
            results = []
            for u in unique_values:
                results.extend(
                    do_count_equivalent_qids(
                        np_arr[np_arr[:, qids_idx[current_qid_idx]] == u],
                        k,
                        qids_idx,
                        current_qid_idx + 1,
                        get_qids,
                    )
                )
            return results

        num_of_vals = np_arr.shape[0]
        if num_of_vals >= k and k != -1:
            return []
        else:
            return (
                [{"qid": np_arr[0, qids_idx].tolist(), "count": num_of_vals}]
                if get_qids is True
                else [num_of_vals]
            )

    return (
        do_count_equivalent_qids(data.values, k, qids_idx, 0, get_qids)
        if isinstance(data, DataFrame)
        else do_count_equivalent_qids(data, k, qids_idx, 0, get_qids)
    )


def is_k_anonymized(data: DataFrame | ndarray, k: int = 2, qids_idx: list = []):
    if len(qids_idx) == 0:
        qids_idx = range(0, data.shape[1])

    def do_is_k_anonymized(np_arr, k, qids_idx, current_qid_idx):
        if current_qid_idx < len(qids_idx):
            unique_values = unique(np_arr.T[qids_idx[current_qid_idx]])
            for u in unique_values:
                result = do_is_k_anonymized(
                    np_arr[np_arr[:, qids_idx[current_qid_idx]] == u],
                    k,
                    qids_idx,
                    current_qid_idx + 1,
                )
                if result is False:
                    return False

            return True

        return np_arr.shape[0] >= k

    return (
        do_is_k_anonymized(data.values, k, qids_idx, 0)
        if isinstance(data, DataFrame)
        else do_is_k_anonymized(data, k, qids_idx, 0)
    )
