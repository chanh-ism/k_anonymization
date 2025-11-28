# +
from abc import ABC, abstractmethod
from enum import Enum

from numpy import arange
from pandas import DataFrame

from k_anonymization.algorithms.type import Algorithm
from k_anonymization.datasets import Dataset

from .utils import get_mean_mode, summarize
from ..utils import generalize_column


# -

class ClusterAnonMethod(Enum):
    SUMMARIZATION = 0
    MEAN_MODE = 1
    GENERALIZATION = 2
    CUSTOM = -1


class ClusteringBasedAlgorithm(Algorithm):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
        anon_method: ClusterAnonMethod = ClusterAnonMethod.SUMMARIZATION,
    ):
        self.anon_method = anon_method
        self.qids_idx = dataset.qids_idx
        self.is_categorical = dataset.is_categorical
        self.hierarchies = dataset.hierarchies
        super().__init__(dataset, k)

    def anonymize(self):
        self.anon_data["__ID"] = arange(self.anon_data.shape[0])
        clusters = self.do_clustering()
        result = []
        for cluster in clusters:
            result.extend(self.anonymize_cluster(cluster))
        self._construct_anon_data(result, columns=(list(self.org_data)+["__ID"]))
        self.anon_data.sort_values("__ID", inplace=True, ignore_index=True)
        self.anon_data.pop("__ID")

    @abstractmethod
    def do_clustering(self):
        pass

    def anonymize_cluster(self, cluster):
        columns = list(zip(*cluster))
        for pos, idx in enumerate(self.qids_idx):
            if self.anon_method == ClusterAnonMethod.SUMMARIZATION:
                columns[idx] = summarize(columns[idx], self.is_categorical[pos])
            elif self.anon_method == ClusterAnonMethod.MEAN_MODE:
                columns[idx] = get_mean_mode(columns[idx], self.is_categorical[pos])
            elif self.anon_method == ClusterAnonMethod.GENERALIZATION:
                level = 0
                while len(set(columns[idx])) > 1:
                    columns[idx] = generalize_column(
                        columns[idx], self.hierarchies[idx], level
                    )
                    level += 1

        return list(zip(*columns))
