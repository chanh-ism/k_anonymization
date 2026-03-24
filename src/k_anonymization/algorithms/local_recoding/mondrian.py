from numpy import median, ndarray
from pandas import factorize, unique

from k_anonymization.core.dataset import Dataset

from .local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)


class ClassicMondrian(LocalRecodingAlgorithm):
    """
    Implementation of Classic Mondrian algorithm.

    Classic Mondrian uses a top-down, greedy domain partitioning
    approach. It divides data into smaller groups by recursively
    splitting the widest-range QID attribute (of the local region at
    each step) on its median value, until there is no longer an
    allowable split satisfying `k`-anonymity (i.e., no possible split
    that provides 2 groups of size ≥ `k`).

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    group_anonymization : GroupAnonymization
        The method to anonymize the resulting groups after applying
        local recoding.
        It is possible to use an example method in
        ``GroupAnonymizationBuiltIn``, or create a custom method
        ``custom_group_anonymization(group: list, props: Any) -> list``.
        Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
    ):
        """
        Initialize Classic Mondrian.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        group_anonymization
            The method to anonymize the resulting groups after applying
            local recoding.
            It is possible to use an example method in
            ``GroupAnonymizationBuiltIn``, or create a custom method
            ``custom_group_anonymization(group: list, props: Any) -> list``.
            Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
        """
        super().__init__(dataset, k, group_anonymization)
        self.__ranges = {}
        self.__cat_uniques = {}
        self.__cat_encodes = {}
        for pos, qid in enumerate(dataset.qids):
            if self.is_categorical[pos]:
                # Store categorical attribute range as the size of unique values
                # Store the original and encoded categorical values
                encode, unique = factorize(self.org_data[qid])
                self.__ranges[self.qids_idx[pos]] = unique.size
                self.__cat_uniques[self.qids_idx[pos]] = unique
                self.__cat_encodes[self.qids_idx[pos]] = encode
            else:
                # Store numerical attribute range as [min, max]
                self.__ranges[self.qids_idx[pos]] = [
                    self.org_data[qid].min(),
                    self.org_data[qid].max(),
                ]

    def sort_qids_idx(self, slice_data: ndarray):
        """
        Determine the order of QID attributes to attempt splitting.

        Heuristically choose the QID attributes with the widest normalized
        range first to minimize information loss. If the normalized ranges
        are tied, pick the attribute with more distinct values.

        Parameters
        ----------
        slice_data : numpy.ndarray
            The records within the current partition.

        Returns
        -------
        list
            Sorted list of QID indices for splitting attempts.
        """

        def get_normalized_range(idx):
            slice_col = slice_data[:, idx]
            # If idx is a categorical attribute
            # (self.__ranges[idx] stores a single integer):
            if isinstance(self.__ranges[idx], int):
                return unique(slice_col).size / self.__ranges[idx]
            else:
                return (slice_col.max() - slice_col.min()) / (
                    self.__ranges[idx][1] - self.__ranges[idx][0]
                )

        return sorted(
            self.qids_idx,
            key=lambda x: (
                -get_normalized_range(x),
                -unique(slice_data[:, x]).size,
            ),
        )

    def __restore_cat_qids_idx(self, slice_data):
        """
        Retore original values of categorical attribute.

        Classic Mondrian processes categorical data as integers;
        this method reverses that mapping before final output.

        Parameters
        ----------
        slice_data : numpy.ndarray
            The records within the current partition.
        """
        for pos, idx in enumerate(self.qids_idx):
            if self.is_categorical[pos]:
                slice_data[:, idx] = self.__cat_uniques[idx][
                    slice_data[:, idx].astype(int)
                ]

    def do_classic_mondrian(self, slice_data: ndarray):
        """
        The recursive core of the Classic Mondrian algorithm.

        Splits a group into two smaller sub-groups at the median
        of the chosen dimension. A split is only accepted if both
        resulting sub-groups contain at least `k` records.

        Parameters
        ----------
        slice_data : numpy.ndarray
            The records within the current partition.

        Returns
        -------
        list
            A list of final partitions.
        """
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

    def do_local_recoding(self):
        """
        Prepare data and initiate the Mondrian partitioning process.

        Factors categorical attributes into integers to allow for
        median-based splitting, then calls the recursive partitioner.

        Returns
        -------
        list
            The collection of partitions (equivalence classes) produced
            by the algorithm.
        """
        np_data = self.anon_data.values
        for pos, idx in enumerate(self.qids_idx):
            if self.is_categorical[pos]:
                # TODO: preserve ordinal order
                np_data[:, idx] = self.__cat_encodes[idx]
        partitions = self.do_classic_mondrian(np_data)
        return partitions
