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
    """
    Implementation of the One-Pass K-Means (OKA) clustering algorithm.

    OKA adopts the idea of the K-Means clustering algorithm. It initiates
    all clusters (groups) of records at once, each with a random seed,
    and distributes the remaining records individually to them
    based on minimal clusters' information losses. Then, OKA
    performs a one-time adjustment step, where furthest records
    in clusters of size > `k` (subject to distance to centroid) are
    picked out and redistributed to those of size < `k`, until every
    cluster contains at least `k` records.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    group_anonymization : GroupAnonymization
        The method to anonymize the resulting clusters after applying
        local recoding.
        It is possible to use an example method in
        ``GroupAnonymizationBuiltIn``, or create a custom method
        ``custom_group_anonymization(group: list, props: Any) -> list``.
        Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
    seed : int
        Random seed for the initial record selection to ensure reproducibility.
    parallel : bool
        Boolean flag to enable parallel processing.
    cpu_cores : int
        The number of CPU cores to utilize when ``parallel`` is True.

    Attributes
    ----------
    is_parallel : bool
        Whether the algorithm is running in parallel mode.
    information_loss : float
        The total information loss calculated across all clusters.
    rand_idx : list
        The indices of the records randomly selected to serve as
        initial cluster seeds.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
        seed: int = None,
        parallel: bool = False,
        cpu_cores: int = Parallel.max_cores - 1,
    ):
        """
        Initialize the OKA algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        group_anonymization : GroupAnonymization
            The method to anonymize the resulting clusters after applying
            local recoding.
            It is possible to use an example method in
            ``GroupAnonymizationBuiltIn``, or create a custom method
            ``custom_group_anonymization(group: list, props: Any) -> list``.
            Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
        seed : int
            Random seed for the initial record selection to ensure reproducibility.
        parallel : bool
            Boolean flag to enable parallel processing.
        cpu_cores : int
            The number of CPU cores to utilize when ``parallel`` is True.
        """
        super().__init__(dataset, k, group_anonymization)
        self.seed = seed
        self.cpu_cores = cpu_cores
        self.is_parallel = parallel
        self.__parallel = Parallel(cpu_cores)
        # Sets up partial functions for distance calculations and cluster
        # initialization, which are crucial for parallel processing.
        self.__get_distance_parallel = partial(oka_get_distance_parallel)
        self.__init_cluster = partial(
            oka_init_cluster,
            qids_idx=self.qids_idx,
            is_categorical=self.is_categorical,
            max_ranges=self.max_ranges,
            hierarchies=self.hierarchies,
        )

    def init_clusters(self):
        """
        Initialize all clusters, each with a random record.

        The number of seeds is calculated as ``round_down(D/k)``,
        where `D` is the total number of records.

        Returns
        -------
        list
            A list of initialized cluster objects.
        """
        random.seed(self.seed)
        self.rand_idx = random.sample(
            range(0, self.anon_data.shape[0]),
            int(self.anon_data.shape[0] / self.k),
        )
        rand_records = [self.anon_data.loc[i].tolist() for i in self.rand_idx]
        return (
            self.__parallel.perform(self.__init_cluster, rand_records)
            if self.is_parallel
            else [self.__init_cluster(r) for r in rand_records]
        )

    def find_best_cluster(self, record: list, clusters: list):
        """
        Find the closest cluster centroid for a given record.

        Calculates the distance between the given record and all current
        cluster centroids to find the best similar cluster.

        Parameters
        ----------
        record : list
            The record to be assigned.
        clusters : list[list]
            The list of clusters.

        Returns
        -------
        int
            The index of the cluster with the minimum distance.
        """
        f = partial(self.__get_distance_parallel, record=record)
        distances = (
            self.__parallel.perform(f, clusters)
            if self.is_parallel
            else [cluster.distance(record) for cluster in clusters]
        )

        best_idx = argmin(distances).item()

        return best_idx

    def get_adjusting_records(self, clusters: list):
        """
        Extract excess records from clusters that exceed size `k`.

        Used during the adjustment stage to free up records that can be
        reassigned to clusters that haven't yet `k`-anonymous.

        Parameters
        ----------
        clusters
            The list of clusters with more than k members.

        Returns
        -------
        list
            A list of records removed from the provided clusters.
        """

        def __get_adjusting_records(cluster, k):
            cluster.sort_by_distance()
            return cluster.remove([0, len(cluster) - k])

        _adjusting_records = [__get_adjusting_records(c, self.k) for c in clusters]
        return sum(_adjusting_records, [])

    def do_local_recoding(self):
        """
        Perform the OKA clustering algorithm.

        The workflow consists of:

        1. Initialize clusters with random seeds.

        2. Clustering stage: Assign every record to the closest cluster
           based on distance to centroid.

        3. Adjustment stage: Rebalance records from "over-full" clusters (`> k`)
           to "under-full" clusters (`< k`) to ensure all clusters are valid.

        Returns
        -------
        list
            The final list of clusters.
        """
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
