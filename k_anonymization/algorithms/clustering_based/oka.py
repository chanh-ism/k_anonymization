# +
import random

from tqdm.auto import tqdm

from k_anonymization.datasets import Dataset

from ..utils import generalize_column
from .type import ClusterAnonMethod, ClusteringBasedAlgorithm
from .utils import get_distance, get_information_loss, get_max_ranges

try:
    __IPYTHON__
    _bar_format = None
except:
    _bar_format = "{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}]"


# -

class OKA_Cluster(object):
    def __init__(
        self,
        first_record,
        qids_idx,
        is_categorical,
        max_ranges,
        hierarchies,
    ):
        self.member = [first_record]
        self.centroid = first_record
        self.qids_idx = qids_idx
        self.is_categorical = is_categorical
        self.max_ranges = max_ranges
        self.hierarchies = hierarchies

    def add(self, record):
        self.member.append(record)
        self.__update_centroid()

    def remove(self, idx: int = 0):
        assert len(self.member) > 0
        self.member.pop(idx)
        self.__update_centroid()

    def remove(self, from_to: list):

        results = self.member[from_to[0] : from_to[1]]
        self.member = self.member[: from_to[0]] + self.member[from_to[1] :]

        self.__update_centroid()
        return results

    def distance(self, record):
        return len(self.member) * get_distance(
            record,
            self.centroid,
            self.qids_idx,
            self.is_categorical,
            self.max_ranges,
            self.hierarchies,
        )

    def sort_by_distance(self):
        self.member.sort(
            key=lambda record: get_distance(
                record,
                self.centroid,
                self.qids_idx,
                self.is_categorical,
                self.max_ranges,
                self.hierarchies,
            )
        )

    def __update_centroid(self):
        if len(self.member) == 0:
            self.centroid = None
        elif len(self.member) == 1:
            self.centroid = self.member[0]
        else:
            centroid = []
            for idx, col in enumerate(zip(*self.member)):
                if idx not in self.qids_idx:
                    centroid.append(-1)
                elif self.is_categorical[self.qids_idx.index(idx)] == True:
                    level = 0
                    values = col[:]
                    while len(set(values)) > 1:
                        values = generalize_column(values, self.hierarchies[idx], level)
                        level += 1
                    centroid.append(values[0])
                else:
                    centroid.append(sum(col) / len(col))
            self.centroid = centroid

    def __getitem__(self, item):
        return self.member[item]

    def __len__(self):
        return len(self.member)


class OKA(ClusteringBasedAlgorithm):

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        anon_method: ClusterAnonMethod = ClusterAnonMethod.SUMMARIZATION,
    ):
        self.hierarchies = dataset.hierarchies
        self.qids_idx = dataset.qids_idx
        self.is_categorical = dataset.is_categorical
        self.max_ranges = get_max_ranges(dataset)
        super().__init__(dataset, k, anon_method)

    def find_best_cluster(self, record, clusters):
        min_distance = float("inf")
        best_idx = None

        for idx, cluster in enumerate(clusters):
            distance = cluster.distance(record)
            if distance < min_distance:
                min_distance = distance
                best_idx = idx

        return best_idx

    def do_clustering(self):
        data = self.anon_data.values.tolist()
        clusters = []

        for i in range(int(len(data) / self.k)):
            r_i_idx = random.randrange(len(data))
            clusters.append(
                OKA_Cluster(
                    data.pop(r_i_idx),
                    self.qids_idx,
                    self.is_categorical,
                    self.max_ranges,
                    self.hierarchies,
                )
            )

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
        adjusting_records = []
        less_than_k_clusters = []
        for cluster in clusters:
            if len(cluster) == self.k:
                adjustment_progress_bar.update(1)
                continue
            elif len(cluster) < self.k:
                less_than_k_clusters.append(cluster)
            else:
                cluster.sort_by_distance()
                adjusting_records.extend(cluster.remove([0, len(cluster) - self.k]))
            adjustment_progress_bar.update(1)

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
