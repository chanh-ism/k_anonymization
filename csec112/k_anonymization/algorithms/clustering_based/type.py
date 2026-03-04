# +
from abc import ABC, abstractmethod

from numpy import arange
from pandas import DataFrame

from k_anonymization.algorithms.type import Algorithm
from k_anonymization.datasets import Dataset

from ..utils import generalize_column
from .utils import get_mean_mode, summarize, get_max_ranges


# -

class ClusterAnonMethod:
    @staticmethod
    def SUMMARIZATION(cluster, props):
        columns = list(zip(*cluster))
        for pos, idx in enumerate(props.qids_idx):
            columns[idx] = summarize(columns[idx], props.is_categorical[pos])
        return list(zip(*columns))

    @staticmethod
    def MEAN_MODE(cluster, props):
        columns = list(zip(*cluster))
        for pos, idx in enumerate(props.qids_idx):
            columns[idx] = get_mean_mode(columns[idx], props.is_categorical[pos])
        return list(zip(*columns))

    @staticmethod
    def GENERALIZATION(cluster, props):
        columns = list(zip(*cluster))
        for pos, idx in enumerate(props.qids_idx):
            level = 0
            while len(set(columns[idx])) > 1:
                columns[idx] = generalize_column(
                    columns[idx], props.hierarchies[idx], level
                )
                level += 1
        return list(zip(*columns))


class ClusteringBasedAlgorithm(Algorithm):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
        cluster_anon_method=ClusterAnonMethod.SUMMARIZATION,
    ):
        self.cluster_anon_method = cluster_anon_method
        self.qids_idx = dataset.qids_idx
        self.is_categorical = dataset.is_categorical
        self.hierarchies = dataset.hierarchies
        self.max_ranges = get_max_ranges(dataset)
        super().__init__(dataset, k)

    def anonymize(self):
        self.anon_data["__ID"] = arange(self.anon_data.shape[0])
        self.clusters = self.do_clustering()
        result = []
        for cluster in self.clusters:
            result.extend(self.cluster_anon_method(cluster, self))
        self._construct_anon_data(result, columns=(list(self.org_data) + ["__ID"]))
        self.anon_data.sort_values("__ID", inplace=True, ignore_index=True)
        self.anon_data.pop("__ID")

    @abstractmethod
    def do_clustering(self):
        pass
