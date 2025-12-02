# +
from numpy import ndarray, median
from pandas import DataFrame, Series, factorize, unique

from k_anonymization import datasets
from k_anonymization.algorithms.clustering_based.type import (
    ClusterAnonMethod,
    ClusteringBasedAlgorithm,
)
from k_anonymization.datasets import Dataset


# -

class ClassicMondrian(ClusteringBasedAlgorithm):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
        cluster_anon_method: ClusterAnonMethod = ClusterAnonMethod.SUMMARIZATION,
    ):
        super().__init__(dataset, k, cluster_anon_method)
        self.__ranges = {}
        self.__cat_uniques = {}
        for pos, qid in enumerate(dataset.qids):
            if self.is_categorical[pos]:
                unique = self.org_data[qid].unique()
                self.__ranges[self.qids_idx[pos]] = unique.size
                self.__cat_uniques[self.qids_idx[pos]] = unique
            else:
                self.__ranges[self.qids_idx[pos]] = [
                    self.org_data[qid].min(),
                    self.org_data[qid].max(),
                ]

    def get_normalized_range(self, slice_col: ndarray, idx):
        def normalized_num():
            return (slice_col.max() - slice_col.min()) / (
                self.__ranges[idx][1] - self.__ranges[idx][0]
            )

        def normalized_cat():
            return unique(slice_col).size / self.__ranges[idx]

        # TODO: Normalized _Ordinal_ Categorical

        return (
            normalized_cat() if isinstance(self.__ranges[idx], int) else normalized_num()
        )

    def sort_qids_idx(self, slice_data: ndarray):
        """
        Sort the qids descending based on
        - the widest normalized range, and
        - the most number of distinct values.
        """
        return sorted(
            self.qids_idx,
            key=lambda x: (
                -self.get_normalized_range(slice_data[:, x], x),
                -unique(slice_data[:, x]).size,
            ),
        )

    def __restore_cat_qids_idx(self, slice_data):
        for pos, idx in enumerate(self.qids_idx):
            if self.is_categorical[pos]:
                slice_data[:, idx] = self.__cat_uniques[idx][
                    slice_data[:, idx].astype(int)
                ]

    def do_classic_mondrian(self, slice_data: ndarray):
        if slice_data.shape[0] < self.k * 2:
            self.__restore_cat_qids_idx(slice_data)
            return [slice_data.tolist()]

        sorted_qids_idx = self.sort_qids_idx(slice_data)
        for idx in sorted_qids_idx:
            median_value = median(slice_data[:, idx])
            lhs = slice_data[slice_data[:, idx] <= median_value]
            rhs = slice_data[slice_data[:, idx] > median_value]

            if lhs.shape[0] < self.k or rhs.shape[0] < self.k:
                continue
            else:
                return self.do_classic_mondrian(lhs) + self.do_classic_mondrian(rhs)
        self.__restore_cat_qids_idx(slice_data)
        return [slice_data.tolist()]

    def do_clustering(self):
        np_data = self.anon_data.values
        for pos, idx in enumerate(self.qids_idx):
            if self.is_categorical[pos]:
                # TODO: preserve ordinal order
                inverse, _ = factorize(np_data[:, idx])
                np_data[:, idx] = inverse
        clusters = self.do_classic_mondrian(np_data)
        return clusters
