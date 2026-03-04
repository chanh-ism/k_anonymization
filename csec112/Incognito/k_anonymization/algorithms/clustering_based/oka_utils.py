from ..utils import generalize_column
from .utils import get_distance


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


def _oka_init_cluster(
    data,
    qids_idx,
    is_categorical,
    max_ranges,
    hierarchies,
):
    return OKA_Cluster(
        data,
        qids_idx,
        is_categorical,
        max_ranges,
        hierarchies,
    )


def _oka_get_distance_parallel(cluster, record):
    return cluster.distance(record)


