# +
from abc import ABC, abstractmethod
from enum import Enum

from pandas import DataFrame

from k_anonymization.algorithms.type import Algorithm
from k_anonymization.datasets import Dataset

from .utils import generalize, get_mean_mode, summarize


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
        super().__init__(dataset, k)

    def anonymize(self):
        clusters = self.do_clustering()
        result = []
        for cluster in clusters:
            result.extend(self.anonymize_cluster(cluster))
        self.anon_data = DataFrame(result, columns=list(self.org_data))

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
                    columns[idx] = generalize(
                        columns[idx], self.hierarchies[idx], level
                    )
                    level += 1

        return list(zip(*columns))
