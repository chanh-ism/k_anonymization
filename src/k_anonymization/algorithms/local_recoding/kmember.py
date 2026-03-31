import random
from functools import partial

from numpy import argmax, argmin
from tqdm.auto import tqdm

from k_anonymization.core import Dataset, Parallel

from ._utils import get_distance, get_information_loss
from .local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)

try:
    __IPYTHON__  # type: ignore # noqa: F821
    _bar_format = None
except:
    _bar_format = "{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}]"


class KMember(LocalRecodingAlgorithm):
    """
    Implementation of the K-Member clustering algorithm.

    K-Member greedily constructs one cluster (group) of records at a time
    until the whole dataset is divided into groups of at least `k` records.
    It initiates the first cluster by randomly selecting an initial record
    (seed), then finds and adds `k-1` other records that minimizes the
    cluster's information loss.
    From the second cluster onward, it picks a new seed which is the furthest
    from the previous seed, and repeats the record selection process until
    there are less than `k` records remaining.
    Finally, each remaining record is added to one of the existing clusters
    that minimizes the cluster's information loss.

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
    seed
        Random seed for the initial record selection to ensure reproducibility.
    parallel
        Boolean flag to enable parallel processing.
    cpu_cores
        The number of CPU cores to utilize when ``parallel`` is True.

    Attributes
    ----------
    is_parallel : bool
        Whether the algorithm is running in parallel mode.
    information_loss : float
        The total information loss calculated across all clusters.

    See Also
    --------
    k_anonymization.core.Parallel :
        Utility wrapper for paralellizing tasks across multiple CPU cores.
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
        Initialize the KMember algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter 'k'.
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
        # Sets up partial functions for distance and information loss
        # calculations, which are crucial for parallel processing.
        self.get_distance = partial(
            get_distance,
            qids_idx=self.qids_idx,
            is_categorical=self.is_categorical,
            max_ranges=self.max_ranges,
            hierarchies=self.hierarchies,
        )
        self.get_information_loss = partial(
            get_information_loss,
            qids_idx=self.qids_idx,
            is_categorical=self.is_categorical,
            max_ranges=self.max_ranges,
            hierarchies=self.hierarchies,
        )

    def find_furthest_record_from_r(self, r: list, data: list):
        """
        Find the most distant record from the given record `r`.

        This is used to find a "seed" for a new cluster that is far
        away from the previously processed cluster.

        Parameters
        ----------
        r : list
            The given record.
        data : list[list]
            The list of available records.

        Returns
        -------
        tuple
            (the furthest record, its index).
        """
        f = partial(self.get_distance, record=r)
        distances = (
            self.__parallel.perform(f, data)
            if self.is_parallel
            else [f(record) for record in data]
        )
        furthest_idx = argmax(distances).item()
        return (data[furthest_idx], furthest_idx)

    def find_best_record(self, data, cluster):
        """
        Find the record that minimizes the information loss of the given cluster.

        Iterates through available records and calculates the potential
        increase in information loss if each record were added to the
        given cluster, then pick the one that causes the lowest loss.

        Parameters
        ----------
        data : list[list]
            The list of available records.
        cluster : list[list]
            The given cluster.

        Returns
        -------
        tuple
            (the best record, its index, the resulting information loss).
        """
        f = partial(self.get_information_loss, cluster=cluster)
        information_losses = (
            self.__parallel.perform(f, data)
            if self.is_parallel
            else [f(record) for record in data]
        )
        best_idx = argmin(information_losses).item()
        return (data[best_idx], best_idx, information_losses[best_idx])

    def find_best_cluster(self, clusters: list, r: list):
        """
        Assign an orphaned record to the most compatible existing cluster.

        Used at the end of the process to assign any remaining records
        to existing clusters while minimizing added information loss.

        Parameters
        ----------
        clusters : list[list]
            The list of already formed clusters.
        r : list
            The record to be assigned.

        Returns
        -------
        tuple
            (index of the best cluster, the resulting information loss).
        """
        information_losses = (
            self.__parallel.perform(
                self.get_information_loss, [r] * len(clusters), clusters
            )
            if self.is_parallel
            else [self.get_information_loss(r, cluster) for cluster in clusters]
        )
        best_idx = argmin(information_losses).item()
        return (best_idx, information_losses[best_idx])

    def do_local_recoding(self):
        """
        Perform the K-Member clustering algorithm.

        The workflow consists of:

        1. Pick a seed record (a random record for the 1st iteration,
           the furthest record from the previous seed otherwise).

        2. Build a cluster of size k by greedily adding records that
           minimize information loss.

        3. Repeat 1-3 until fewer than k records remain.

        4. Distribute remaining records to the most suitable existing
           clusters.

        Returns
        -------
        list
            The final list of clusters.
        """
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
