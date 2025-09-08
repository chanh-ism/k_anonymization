# +
from numpy import argmax

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

    def pick_attribute(self, df, qids):
        # pick_attribute_with_highest_cardinality
        cardinalities = [len(df[x].unique().tolist()) for x in qids]
        return qids[argmax(cardinalities)]

    def anonymize(self):
        qids = self.dataset.qids
        hierarchies_tracking = {}

        while True:
            if self.suppression_threshold == 0 and is_k_anonymized(
                self.anon_data, self.k, qids
            ):
                break

            else:
                not_k_anonymized_qids = find_not_k_anonymized_qids(
                    self.anon_data, self.k, qids
                )
                if (
                    sum([x["count"] for x in not_k_anonymized_qids])
                    <= self.suppression_threshold
                ):
                    self.suppressed_qids = not_k_anonymized_qids
                    for qid in not_k_anonymized_qids:
                        queries = []
                        for i, att in enumerate(qid["qid"]):
                            if att == "*":
                                continue
                            if type(att) is str:
                                queries.append(f'`{qids[i]}` != "{att}"')
                            else:
                                queries.append(f"`{qids[i]}` != {att}")
                        self.anon_data.query(" or ".join(queries), inplace=True)

                    break

            generalized_att = self.pick_attribute(self.anon_data, qids)
            if generalized_att in list(hierarchies_tracking):
                hierarchies_tracking[generalized_att] = (
                    hierarchies_tracking[generalized_att] + 1
                )
            else:
                hierarchies_tracking[generalized_att] = 0
            generalize(
                self.anon_data,
                self.dataset.hierarchies[generalized_att],
                hierarchies_tracking[generalized_att],
            )
        return self.anon_data
