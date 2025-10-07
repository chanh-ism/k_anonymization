# +
import random

from tqdm.auto import tqdm

from k_anonymization.datasets import Dataset

from .type import ClusterAnonMethod, ClusteringBasedAlgorithm
from .utils import get_distance, get_information_loss, get_max_ranges


# -

class KMember(ClusteringBasedAlgorithm):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
        anon_method: ClusterAnonMethod = ClusterAnonMethod.SUMMARIZATION,
    ):
        self.hierarchies = dataset.hierarchies
        self.qids_idx = dataset.qids_idx
        self.is_categorical = dataset.is_categorical
        super().__init__(dataset, k, anon_method)

    def find_furthest_record_from_r(self, r, data):
        max_distance = 0
        furthest_idx = None
        furthest_r = None

        for idx, record in enumerate(data):
            this_distance = get_distance(
                r,
                record,
                self.qids_idx,
                self.is_categorical,
                self.max_ranges,
                self.hierarchies,
            )
            if this_distance > max_distance:
                max_distance = this_distance
                furthest_r = record
                furthest_idx = idx

        return (furthest_r, furthest_idx)

    def find_best_record(self, data, cluster):
        min_information_loss = float("inf")
        best_idx = None
        best_r = None

        for idx, record in enumerate(data):
            this_information_loss = get_information_loss(
                record,
                cluster,
                self.qids_idx,
                self.is_categorical,
                self.max_ranges,
                self.hierarchies,
            )
            if this_information_loss < min_information_loss:
                min_information_loss = this_information_loss
                best_r = record
                best_idx = idx

        return (best_r, best_idx, min_information_loss)

    def find_best_cluster(self, clusters, r):
        min_information_loss = float("inf")
        best_idx = None

        for idx, cluster in enumerate(clusters):
            this_information_loss = get_information_loss(
                r,
                cluster,
                self.qids_idx,
                self.is_categorical,
                self.max_ranges,
                self.hierarchies,
            )
            if this_information_loss < min_information_loss:
                min_information_loss = this_information_loss
                best_idx = idx

        return (best_idx, min_information_loss)

    def do_clustering(self):
        data = self.anon_data.values.tolist()
        self.max_ranges = get_max_ranges(
            data,
            self.qids_idx,
            self.is_categorical,
            self.hierarchies,
        )
        clusters = []
        information_losses = []
        r_i = None

        progress_bar = tqdm(
            total=len(data),
            desc="   Clustering Progress",
            bar_format="{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}]",
        )

        while len(data) >= self.k:
            if r_i is None:
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

        return clusters


