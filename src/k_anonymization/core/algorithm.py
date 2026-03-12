from abc import ABC, abstractmethod

from k_anonymization.core.dataset import Dataset, ITableDF


class Algorithm(ABC):
    """
    Abstract class for implementing k-anonymization algorithms.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter 'k'.

    Attributes
    ----------
    k : int
        The privacy parameter 'k'.
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    org_data : ItableDF
        The original data extracted from `dataset`.
    anon_data : ITableDF or None
        The anonymized data.
    suppressed_qids : list
        The records suppressed (removed) from the original data.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
    ):
        """
        Initialize the Algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter 'k'.
        """
        self.k = k
        self.dataset = dataset
        self.anon_data = None
        self.suppressed_qids = None

    @property
    def org_data(self):
        return self.dataset.df

    def __getattribute__(self, name):
        """
        Clears previous anonymization results when calling `anonymize`.
        """
        if name == "anonymize":
            self.anon_data = self.org_data[:]
            self.suppressed_qids = None
        return object.__getattribute__(self, name)

    @abstractmethod
    def anonymize(self):
        """
        The core anonymization logic to be implemented by subclasses.

        This method should transform `self.anon_data` such that
        it satisfies the k-anonymity requirement.
        """
        pass

    def _construct_anon_data(self, data, columns, table_name="Anonymized Data"):
        """
        Helper method to finalize and wrap the anonymized data.

        Converts anonymized data into an `ITableDF` object for
        standardized display and further analysis.

        Parameters
        ----------
        data : array-like
            The anonymized data.
        columns : list
            The list of column names for the new dataframe.
        table_name : str, optional
            A descriptive label for the resulting table.
        """
        self.anon_data = ITableDF(data, columns=columns, table_name=table_name)
