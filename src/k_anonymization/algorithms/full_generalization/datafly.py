from numpy import argmax, unique

from k_anonymization.core import Algorithm, Dataset
from k_anonymization.evaluation.anonymity import (
    find_not_k_anonymous_qids,
    is_k_anonymous,
)

from ..utils import generalize_column


class Datafly(Algorithm):
    """
    Implementation of Datafly algorithm.

    Datafly applies an iterative heuristic to generalize attributes with
    the highest cardinality (number of unique values) until the dataset
    satisfies `k`-anonymity.

    If an optional suppression threshold is set, Datafly generalizes the
    dataset until (entire dataset - suppression threshold) satisfies
    `k`-anonymity.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    suppression_threshold : int, default 0
         The number of allowed suppressed records.

        The maximum number of records that can be removed (suppressed)
        from the dataset to satisfy `k`-anonymity.

    Attributes
    ----------
    suppression_threshold : int
        The number of allowed suppressed records.
    hierarchies_tracking : dict
        A mapping of attribute names to their current generalization
        level in the hierarchy.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        suppression_threshold: int = 0,
    ):
        """
        Initialize the Datafly algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        suppression_threshold : int, optional
            The number of allowed suppressed records.
        """
        self.suppression_threshold = suppression_threshold
        super().__init__(dataset, k)

    def pick_attribute(self, np_data, qids_idx, qids):
        """
        Pick the attribute with the highest cardinality.

        This heuristic is used to decide which attribute to generalize next,
        aiming to reduce the uniqueness of records as quickly as possible.

        Parameters
        ----------
        np_data : numpy.ndarray
            The current state of the data in a NumPy array format.
        qids_idx : list
            The column indices of the Quasi-Identifiers.
        qids : list
            The names of the Quasi-Identifiers.

        Returns
        -------
        int
            The column index of the attribute with the highest cardinality.
        str
            The name of the attribute with the highest cardinality.
        """
        _np_data = np_data.T
        cardinalities = [len(unique(_np_data[idx])) for idx in qids_idx]
        max_cardinality = argmax(cardinalities)
        return qids_idx[max_cardinality], qids[max_cardinality]

    def anonymize(self):
        """
        Run the Datafly algorithm.

        Iteratively generalizes the attribute with the highest cardinality.
        If the data state does not satisfy `k`-anonymity but the number of
        outlying records is below the `suppression_threshold`,
        those records are removed to finalize anonymization.
        """
        qids = self.dataset.qids
        qids_idx = self.dataset.qids_idx
        np_anon_data = self.anon_data.values
        self.hierarchies_tracking = dict.fromkeys(qids, 0)

        while True:
            if self.suppression_threshold == 0 and is_k_anonymous(
                np_anon_data, self.k, qids_idx
            ):
                break

            else:
                not_k_anonymized_qids = find_not_k_anonymous_qids(
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
                                condition |= np_anon_data[:, qids_idx[i]] != att
                        np_anon_data = np_anon_data[condition]

                    break

            generalized_att_idx, generalized_att = self.pick_attribute(
                np_anon_data, qids_idx, qids
            )
            generalized_values, is_suppressed = generalize_column(
                np_anon_data[:, generalized_att_idx],
                self.dataset.hierarchies[generalized_att],
                self.hierarchies_tracking[generalized_att],
                self.hierarchies_tracking[generalized_att] + 1,
            )
            np_anon_data[:, generalized_att_idx] = generalized_values

            if is_suppressed:
                self.hierarchies_tracking[generalized_att] = -1
            else:
                self.hierarchies_tracking[generalized_att] += 1

        self._construct_anon_data(np_anon_data, columns=list(self.anon_data))
