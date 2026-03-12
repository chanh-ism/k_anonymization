import random
from functools import partial

from numpy import argmin
from tqdm.auto import tqdm

from k_anonymization.core.dataset import Dataset
from k_anonymization.utils.parallel import Parallel

from .._utils import get_information_loss
from ..local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)
from ._utils import oka_get_distance_parallel, oka_init_cluster

try:
    __IPYTHON__  # type: ignore # noqa: F821
    _bar_format = None
except:
    _bar_format = "{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}]"


class OKA(LocalRecodingAlgorithm):

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        cluster_anon_method: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
        seed: int = None,
        parallel: bool = False,
        cpu_cores: int = Parallel.max_cores - 1,
    ):
        super().__init__(dataset, k, cluster_anon_method)
        self.seed = seed
        self.cpu_cores = cpu_cores
        self.is_parallel = parallel
        self.__parallel = Parallel(cpu_cores)
        self.__get_distance_parallel = partial(oka_get_distance_parallel)
        self.__init_cluster = partial(
            oka_init_cluster,
            qids_idx=self.qids_idx,
            is_categorical=self.is_categorical,
            max_ranges=self.max_ranges,
            hierarchies=self.hierarchies,
        )

    def find_best_cluster(self, record, clusters):
        f = partial(self.__get_distance_parallel, record=record)
        distances = (
            self.__parallel.perform(f, clusters)
            if self.is_parallel
            else [cluster.distance(record) for cluster in clusters]
        )

        best_idx = argmin(distances).item()

        return best_idx

    def init_clusters(self):
        rand_records = [self.anon_data.loc[i].tolist() for i in self.rand_idx]
        return (
            self.__parallel.perform(self.__init_cluster, rand_records)
            if self.is_parallel
            else [self.__init_cluster(r) for r in rand_records]
        )

    def get_adjusting_records(self, clusters):
        def __get_adjusting_records(cluster, k):
            cluster.sort_by_distance()
            return cluster.remove([0, len(cluster) - k])

        _adjusting_records = [__get_adjusting_records(c, self.k) for c in clusters]
        return sum(_adjusting_records, [])

    def do_local_recoding(self):
        random.seed(self.seed)
        self.rand_idx = random.sample(
            range(0, self.anon_data.shape[0]),
            int(self.anon_data.shape[0] / self.k),
        )

        if self.is_parallel:
            print(f"Parallelize with {self.cpu_cores} core(s).")
            self.__parallel.activate()

        clusters = self.init_clusters()

        self.anon_data.drop(self.rand_idx, inplace=True)
        data = self.anon_data.values.tolist()

        clustering_progress_bar = tqdm(
            total=len(data),
            desc="   Clustering Progress",
            bar_format=_bar_format,
        )

        # Clustering Stage
        while len(data) > 0:
            record = data.pop()
            best_cluster_idx = self.find_best_cluster(record, clusters)
            clusters[best_cluster_idx].add(record)
            clustering_progress_bar.update(1)

        clustering_progress_bar.close()
        adjustment_progress_bar = tqdm(
            total=(len(clusters) * 2),
            desc="   Adjustment Progress",
            bar_format=_bar_format,
        )

        # Adjustment Stage
        less_than_k_clusters = []
        more_than_k_clusters = []
        for cluster in clusters:
            adjustment_progress_bar.update(1)
            if len(cluster) == self.k:
                continue
            elif len(cluster) < self.k:
                less_than_k_clusters.append(cluster)
            else:
                more_than_k_clusters.append(cluster)
        adjusting_records = self.get_adjusting_records(more_than_k_clusters)

        while len(adjusting_records) > 0:
            record = adjusting_records.pop()
            if len(less_than_k_clusters) > 0:
                best_cluster_idx = self.find_best_cluster(record, less_than_k_clusters)
                less_than_k_clusters[best_cluster_idx].add(record)
                if len(less_than_k_clusters[best_cluster_idx]) >= self.k:
                    less_than_k_clusters.pop(best_cluster_idx)
            else:
                best_cluster_idx = self.find_best_cluster(record, clusters)
                clusters[best_cluster_idx].add(record)

        adjustment_progress_bar.update(len(clusters))
        adjustment_progress_bar.close()

        self.information_loss = 0
        for cluster in clusters:
            self.information_loss += get_information_loss(
                None,
                cluster.member,
                self.qids_idx,
                self.is_categorical,
                self.max_ranges,
                self.hierarchies,
            )

        return clusters
