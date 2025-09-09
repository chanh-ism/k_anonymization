# +
from numpy import argmax, unique
from pandas import DataFrame

from k_anonymization.evaluation.utils import find_not_k_anonymized_qids, is_k_anonymized

from ...datasets import Dataset
from ..type import Algorithm
from ..utils import generalize


# -

class Datafly(Algorithm):

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        suppression_threshold: int = 0,
    ):
        self.suppression_threshold = suppression_threshold
        super().__init__(dataset, k)

    def pick_attribute(self, np_data, qids_idx, qids):
        # pick_attribute_with_highest_cardinality
        _np_data = np_data.T
        cardinalities = [len(unique(_np_data[idx])) for idx in qids_idx]
        max_cardinality = argmax(cardinalities)
        return qids_idx[max_cardinality], qids[max_cardinality]

    def anonymize(self):
        qids = self.dataset.qids
        qids_idx = self.dataset.qids_idx
        np_anon_data = self.anon_data.values
        hierarchies_tracking = {}

        while True:
            if self.suppression_threshold == 0 and is_k_anonymized(
                np_anon_data, self.k, qids_idx
            ):
                break

            else:
                not_k_anonymized_qids = find_not_k_anonymized_qids(
                    np_anon_data, self.k, qids_idx
                )
                if (
                    sum([x["count"] for x in not_k_anonymized_qids])
                    <= self.suppression_threshold
                ):
                    self.suppressed_qids = not_k_anonymized_qids
                    for qid in not_k_anonymized_qids:
                        condition = [False] * len(np_anon_data)
                        for i, att in enumerate(qid["qid"]):
                            if att == "*":
                                continue
                            else:
                                condition |= np_anon_data[0, qids_idx[i]] != att
                        np_anon_data = np_anon_data[condition]

                    break

            generalized_att_idx, generalized_att = self.pick_attribute(
                np_anon_data, qids_idx, qids
            )
            if generalized_att in list(hierarchies_tracking):
                hierarchies_tracking[generalized_att] = (
                    hierarchies_tracking[generalized_att] + 1
                )
            else:
                hierarchies_tracking[generalized_att] = 0
            generalize(
                np_anon_data,
                self.dataset.hierarchies[generalized_att],
                generalized_att_idx,
                hierarchies_tracking[generalized_att],
            )

        self.anon_data = DataFrame(np_anon_data, columns=list(self.anon_data))
        return self.anon_data
