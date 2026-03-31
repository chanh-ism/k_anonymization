__all__ = ["GroupAnonymization", "GroupAnonymizationBuiltIn", "LocalRecodingAlgorithm"]

from abc import abstractmethod
from typing import Callable, Collection

from numpy import arange

from k_anonymization.core.algorithm import Algorithm
from k_anonymization.core.dataset import Dataset

from ._utils import get_max_ranges

type GroupAnonymization = Callable[
    [Collection[Collection], dict], Collection[Collection]
]
"""
Prototype of a group anonymization method for local recoding algorithms.

Local recoding algorithms typically split the original records into groups.
Then, a group anonymization method (implementation of this type) is applied 
to make records become indistinguishable in their respective group.

Parameters
----------
group : Collection[Collection]
    The group of records to be anonymized.
props : dict
    A dictionary containing necessary properties for anonymization.

Returns
-------
Collection[Collection]
    The anonymized group.

See Also
--------
LocalRecodingAlgorithm : 
    Abstract class for local recoding-based k-anonymization algorithms.
GroupAnonymizationBuiltIn :
    A set of built-in ``GroupAnonymization``.
"""


class GroupAnonymizationBuiltIn:
    """
    A set of built-in ``GroupAnonymization``.
    """

    @staticmethod
    def SUMMARIZATION(group: Collection[Collection], props: dict):
        """
        Anonymize a group by creating a summary range or set.

        Numerical values are converted into intervals (e.g., "20-30"),
        while categorical values are listed as a set.

        Parameters
        ----------
        group : Collection[Collection]
            A group of records to be anonymized.
        props : dict
            Properties dict containing `qids_idx` and `is_categorical`.

        Returns
        -------
        Collection[Collection]
            The anonymized group.
        """

        def summarize(values, is_cat):
            anon_value = None
            if is_cat:
                try:
                    anon_value = " & ".join(set(values))
                except:
                    anon_value = " & ".join([str(x) for x in set(values)])
            else:
                if len(set(values)) == 1:
                    anon_value = f"{values[0]}"
                else:
                    anon_value = f"{min(values)} ~ {max(values)}"
            return list(map(lambda x: anon_value, values))

        columns = list(zip(*group))
        for pos, idx in enumerate(props.qids_idx):
            columns[idx] = summarize(columns[idx], props.is_categorical[pos])
        return list(zip(*columns))

    @staticmethod
    def MEAN_MODE(group: Collection[Collection], props: dict):
        """
        Anonymize a group by mean and mode.

        Replace numerical QID values with the group mean and
        categorical QID values with the group mode (most frequent value).

        Parameters
        ----------
        group : Collection[Collection]
            A group of records to be anonymized.
        props : dict
            Properties dict containing `qids_idx` and `is_categorical`.

        Returns
        -------
        Collection[Collection]
            The anonymized group.
        """

        def get_mean_mode(values, is_cat):
            anon_value = None
            if is_cat:
                anon_value = max(values, key=values.count)
            else:
                anon_value = sum(values) / len(values)
            return list(map(lambda x: anon_value, values))

        columns = list(zip(*group))
        for pos, idx in enumerate(props.qids_idx):
            columns[idx] = get_mean_mode(columns[idx], props.is_categorical[pos])
        return list(zip(*columns))

    @staticmethod
    def GENERALIZATION(group: Collection[Collection], props: dict):
        """
        Anonymize a group by full-(sub)domain generalization.

        For each QID attribute, generalize its values in all records until
        they are identical.

        Parameters
        ----------
        group : Collection[Collection]
            A group of records to be anonymized.
        props : dict
            Properties dict containing `qids_idx` and `hierarchies`.

        Returns
        -------
        Collection[Collection]
            The anonymized group.
        """
        columns = list(zip(*group))
        for _, idx in enumerate(props.qids_idx):
            columns[idx] = [
                props.hierarchies[idx].get_lowest_common_ancestor(columns[idx])
            ] * len(columns[idx])
        return list(zip(*columns))


class LocalRecodingAlgorithm(Algorithm):
    """
    Abstract class for local recoding-based k-anonymization algorithms.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter 'k'.
    group_anonymization : GroupAnonymization
        The method to anonymize the resulting groups after applying
        local recoding.
        It is possible to use an example method in
        `GroupAnonymizationBuiltIn`, or create a custom method
        `custom_group_anonymization(group: list, props: Any) -> list`.
        Default: `GroupAnonymizationBuiltIn.SUMMARIZATION`

    Attributes
    ----------
    [...] : attributes inherited from ``Algorithm``

    group_anonymization : GroupAnonymization
        Group anonymization method.
    groups : list | None
        List of groups of *original* records after applying local recoding
        (before applying the group anonymization method)
    max_ranges : list[int | None]
        Maximum range (or unique count) for each QID attribute.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        group_anonymization: GroupAnonymization = GroupAnonymizationBuiltIn.SUMMARIZATION,
    ):
        """
        Initialize the local recoding algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter 'k'.
        group_anonymization : GroupAnonymization
            The method to anonymize the resulting groups after applying
            local recoding.
            It is possible to use an example method in
            ``GroupAnonymizationBuiltIn``, or create a custom method
            ``custom_group_anonymization(group: list, props: Any) -> list``.
            Default: ``GroupAnonymizationBuiltIn.SUMMARIZATION``
        """
        self.groups = None
        self.group_anonymization = group_anonymization
        self.qids_idx = dataset.qids_idx
        self.is_categorical = dataset.is_categorical
        self.hierarchies = dataset.hierarchies
        self.max_ranges = get_max_ranges(dataset)
        super().__init__(dataset, k)

    def anonymize(self):
        """
        Execute the local recoding anonymization workflow.

        The process follows these steps:

        1. Inject temporary IDs to track original row ordering.

        2. Split data into groups using `do_local_recoding`.

        3. Apply the group anonymization strategy to each group.

        4. Reconstruct the dataset and restore original row order.

        5. Clean up temporary IDs.
        """
        self.anon_data["__ID"] = arange(self.anon_data.shape[0])
        self.groups = self.do_local_recoding()
        result = []
        for group in self.groups:
            result.extend(self.group_anonymization(group, self))
        self._construct_anon_data(result, columns=(list(self.org_data) + ["__ID"]))
        self.anon_data.sort_values("__ID", inplace=True, ignore_index=True)
        self.anon_data.pop("__ID")

    @abstractmethod
    def do_local_recoding(self):
        """
        Split the original data into groups of records.

        This abstract method must be implemented by subclasses to define
        how the data is split into groups that satisfy k-anonymity.

        Returns
        -------
        list
            A list of groups, where each group contains records from the
            original data
        """
        pass
