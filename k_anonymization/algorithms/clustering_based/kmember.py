# +
import random
import time
from typing import Literal

from pandas import DataFrame
from tqdm.auto import tqdm

from k_anonymization.algorithms.type import Algorithm
from k_anonymization.datasets import Dataset

from .utils import *


# -

class KMember(Algorithm):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
        anon_method: Literal[
            "summarization", "generalization", "mean_mode", "custom"
        ] = "summarization",
    ):
        self.hierarchies = dataset.hierarchies
        self.qids_idx = dataset.qids_idx
        self.is_categorical = dataset.is_categorical
        self.anon_method = anon_method
        super().__init__(dataset, k)

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
                self.num_ranges,
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
                self.num_ranges,
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
                self.num_ranges,
                self.hierarchies,
            )
            if this_information_loss < min_information_loss:
                min_information_loss = this_information_loss
                best_idx = idx

        return (best_idx, min_information_loss)

    def do_clustering_kmember(self, data):
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

        return (clusters, sum(information_losses))

    def anonymize_cluster(self, cluster):
        columns = list(zip(*cluster))
        for pos, idx in enumerate(self.qids_idx):
            if self.anon_method == "generalization":
                level = 0
                while len(set(columns[idx])) > 1:
                    columns[idx] = generalize(columns[idx], self.hierarchies[idx], level)
                    level += 1
            elif self.anon_method == "mean_mode":
                columns[idx] = get_mean_mode(columns[idx], self.is_categorical[pos])
            elif self.anon_method == "summarization":
                columns[idx] = summarize(columns[idx], self.is_categorical[pos])

        return list(zip(*columns))

    def anonymize(self):
        data = self.anon_data.values.tolist()[0:1000]
        self.num_ranges = get_num_ranges(
            data,
            self.qids_idx,
            self.is_categorical,
        )
        result = []
        clusters, self.information_loss = self.do_clustering_kmember(data)

        progress_bar = tqdm(
            total=len(clusters),
            desc="Anonymization Progress",
            bar_format="{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}]",
        )

        for cluster in clusters:
            result.extend(self.anonymize_cluster(cluster))

        progress_bar.close()
        self.anon_data = DataFrame(result, columns=list(self.anon_data))


