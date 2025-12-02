# +
import random
from functools import partial

from numpy import argmax, argmin
from tqdm.auto import tqdm

from k_anonymization.datasets import Dataset
from k_anonymization.utils.parallel import Parallel

from .type import ClusterAnonMethod, ClusteringBasedAlgorithm
from .utils import get_distance, get_information_loss, get_max_ranges

try:
    __IPYTHON__
    _bar_format = None
except:
    _bar_format = "{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}]"


# -

class KMember(ClusteringBasedAlgorithm):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
        cluster_anon_method: ClusterAnonMethod = ClusterAnonMethod.SUMMARIZATION,
        seed: int = None,
        parallel: bool = False,
        cpu_cores: int = Parallel.max_cores - 1,
    ):
        super().__init__(dataset, k, cluster_anon_method)
        self.seed = seed
        self.cpu_cores = cpu_cores
        self.is_parallel = parallel
        self.__parallel = Parallel(cpu_cores)
        max_ranges = get_max_ranges(dataset)
        self.get_distance = partial(
            get_distance,
            qids_idx=self.qids_idx,
            is_cat=self.is_categorical,
            max_ranges=max_ranges,
            hierarchies=self.hierarchies,
        )
        self.get_information_loss = partial(
            get_information_loss,
            qids_idx=self.qids_idx,
            is_cat=self.is_categorical,
            max_ranges=max_ranges,
            hierarchies=self.hierarchies,
        )

    def find_furthest_record_from_r(self, r, data):
        f = partial(self.get_distance, record=r)
        distances = (
            self.__parallel.perform(f, data)
            if self.is_parallel
            else [f(record) for record in data]
        )
        furthest_idx = argmax(distances).item()
        return (data[furthest_idx], furthest_idx)

    def find_best_record(self, data, cluster):
        f = partial(self.get_information_loss, cluster=cluster)
        information_losses = (
            self.__parallel.perform(f, data)
            if self.is_parallel
            else [f(record) for record in data]
        )
        best_idx = argmin(information_losses).item()
        return (data[best_idx], best_idx, information_losses[best_idx])

    def find_best_cluster(self, clusters, r):
        information_losses = (
            self.__parallel.perform(
                self.get_information_loss, [r] * len(clusters), clusters
            )
            if self.is_parallel
            else [self.get_information_loss(r, cluster) for cluster in clusters]
        )
        best_idx = argmin(information_losses).item()
        return (best_idx, information_losses[best_idx])

    def do_clustering(self):
        data = self.anon_data.values.tolist()

        clusters = []
        information_losses = []
        r_i = None

        if self.is_parallel:
            print(f"Parallelize with {self.cpu_cores} core(s).")
            self.__parallel.activate()

        progress_bar = tqdm(
            total=len(data),
            desc="   Clustering Progress",
            bar_format=_bar_format,
        )

        while len(data) >= self.k:
            if r_i is None:
                random.seed(self.seed)
                r_i_idx = random.randrange(len(data))
                r_i = data[r_i_idx]
            else:
                r_i, r_i_idx = self.find_furthest_record_from_r(r_i, data)
            data.pop(r_i_idx)
            this_cluster = [r_i]
            this_information_loss = None

            while len(this_cluster) < self.k:
                r_j, r_j_idx, this_information_loss = self.find_best_record(
                    data, this_cluster
                )
                data.pop(r_j_idx)
                this_cluster.append(r_j)

            information_losses.append(this_information_loss)
            clusters.append(this_cluster)
            progress_bar.update(self.k)

        for r in data:
            cluster_idx, information_loss = self.find_best_cluster(clusters, r)
            information_losses[cluster_idx] = information_loss
            clusters[cluster_idx].append(r)
            progress_bar.update(1)

        self.information_loss = sum(information_losses)
        progress_bar.close()

        if self.is_parallel:
            self.__parallel.deactivate()

        return clusters


