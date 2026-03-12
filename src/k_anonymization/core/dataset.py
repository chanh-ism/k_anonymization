import json
from functools import cached_property
from os import path as os_path

from itables import show as itables_show
from pandas import DataFrame, read_csv

from k_anonymization import __file__ as top_dir

from .frame import ITableDF
from .hierarchy import HierarchiesDict


class Dataset:
    """
    Management class for handling dataset and its properties.

    This class serves as a central hub for a dataset, orchestrating
    the loading of raw data, attribute properties, and generalization
    hierarchies.

    Parameters
    ----------
    name : str
        The name of the dataset. This should correspond to a directory
        in `k_anonymization/datasets/`.

    Attributes
    ----------
    all_datasets : list
        A class-level list containing all instantiated Dataset objects.
    name : str
        The name assigned to the dataset instance.
    """

    # TODO: implement classes for QIDs

    all_datasets = []

    def __init__(self, name: str):
        """
        Initializes the Dataset instance, and registers it in the global
        `all_datasets` list.

        Parameters
        ----------
        name : str
            The name of the dataset, which corresponds to a directory in
            `k_anonymization/datasets/`.
        """
        self.name = name
        self.__df = None
        self.__hierarchies = None
        self.__props = None
        Dataset.all_datasets.append(self)

    def __str__(self):
        """
        Return the uppercase name of the dataset.

        Returns
        -------
        str
            The dataset name in all caps.
        """
        return self.name.upper()

    @cached_property
    def path(self):
        """
        Get the absolute file path to the dataset directory.

        Returns
        -------
        str
            The directory path where the dataset assets are stored.
        """
        return f"{os_path.dirname(top_dir)}/datasets/{self.name}"

    @cached_property
    def props(self):
        """
        Load and cache the dataset properties from a JSON file.

        The file is expected to be located at `path/props.json`.

        Returns
        -------
        dict
            A dictionary containing metadata such as QID indices and
            categorical flags.
        """
        if self.__props is None:
            with open(f"{self.path}/props.json") as f:
                _props = json.load(f)
            self.__props = _props
        return self.__props

    @cached_property
    def qids(self):
        """
        Retrieve the names of the Quasi-Identifier (QID) columns.

        Returns
        -------
        list
            A list of column names designated as QIDs.
        """
        return [self.df.columns[x] for x in self.props["qi_index"]]

    @cached_property
    def qids_idx(self):
        """
        Retrieve the indices of the Quasi-Identifier (QID) columns.

        Returns
        -------
        list
            A list of integers representing the column positions of QIDs.
        """
        return self.props["qi_index"]

    @cached_property
    def is_categorical(self):
        """
        Check which QID attributes are categorical.

        Returns
        -------
        list
            A list of booleans indicating if the QID at that position
            is categorical.
        """
        return self.props["is_category"]

    @cached_property
    def target(self):
        """
        Retrieve the name of the target/sensitive attribute.

        Returns
        -------
        str
            The column name of the target attribute.
        """
        return self.props["target"]

    @cached_property
    def qids_categorial(self):
        """
        Filter and return categorical QID names.

        Returns
        -------
        list
            Names of QIDs that are categorical.
        """
        return [qid for pos, qid in enumerate(self.qids) if self.is_categorical[pos]]

    @cached_property
    def qids_numerical(self):
        """
        Filter and return numerical QID names.

        Returns
        -------
        list
            Names of QIDs that are numerical.
        """
        return [
            qid for pos, qid in enumerate(self.qids) if not self.is_categorical[pos]
        ]

    @cached_property
    def qids_idx_categorial(self):
        """
        Filter and return categorical QID indices.

        Returns
        -------
        list
            Column indices of QIDs that are categorical.
        """
        return [
            idx for pos, idx in enumerate(self.qids_idx) if self.is_categorical[pos]
        ]

    @cached_property
    def qids_idx_numerical(self):
        """
        Filter and return numerical QID indices.

        Returns
        -------
        list
            Column indices of QIDs that are numerical.
        """
        return [
            idx for pos, idx in enumerate(self.qids_idx) if not self.is_categorical[pos]
        ]

    # TODO: Update other modules to use
    # - qids_categorial
    # - qids_numerical
    # - qids_idx_categorial
    # - qids_idx_numerical

    @cached_property
    def info(self):
        """
        Generate a summary DataFrame of the dataset's attributes.

        Provides a breakdown of each column's status as a QID, numerical,
        or categorical attribute, along with the count of unique values.

        Returns
        -------
        pandas.DataFrame
            A summary table with metadata labels as the index.
        """
        _info_all = {}
        for col_idx, col in enumerate(list(self.df)):
            if col_idx not in self.qids_idx:
                _info = ["", "", ""]
            else:
                _info = [
                    "o" if col_idx in self.qids_idx else "",
                    (
                        "o"
                        if self.is_categorical[self.qids_idx.index(col_idx)] is False
                        else ""
                    ),
                    (
                        "o"
                        if self.is_categorical[self.qids_idx.index(col_idx)] is True
                        else ""
                    ),
                ]
            _info.append(self.df[col].unique().size)
            _info_all[col] = _info
        return DataFrame(
            _info_all,
            index=[
                "Quasi-Identifier (QID)",
                "Numerical Attribute",
                "Categorical Attribute",
                "No. of Unique Values",
            ],
        )

    @property
    def df(self):
        """
        Lazy-load the dataset as a ITableDF.

        Returns
        -------
        ITableDF
            The underlying data stored in the CSV file.
        """
        if self.__df is None:
            self.reload_df()
        return self.__df

    @property
    def hierarchies(self):
        """
        Lazy-load the generalization hierarchies for this dataset.

        Returns
        -------
        HierarchiesDict
            A manager containing hierarchy definitions for QID columns.
        """
        if self.__hierarchies is None:
            self.__hierarchies = HierarchiesDict(
                f"{self.path}/hierarchies",
                len(list(self.df)),
                self.qids,
                self.qids_idx,
                self.df[0:1].values[0].tolist(),
            )
        return self.__hierarchies

    def reload_df(self):
        """
        Read the CSV file of the Dataset from disk.
        """
        self.__df = ITableDF(
            read_csv(f"{self.path}/{self.name}.csv"),
            table_name=str(self),
        )

    def describe(self):
        """
        Render interactive dataframe showing dataset info.
        """
        itables_show(
            DataFrame(
                [
                    str(self),
                    self.df.shape[0],
                    self.df.shape[1],
                ],
                index=[
                    "Dataset Name",
                    "No. of Records",
                    "No. of Attributes",
                ],
            ),
            scrollX=True,
            ordering={"indicators": False},
            columnDefs=[
                {
                    "className": "no-header",
                    "target": "_all",
                }
            ],
        )

        itables_show(
            self.info,
            fixedColumns={"start": 1},
            scrollX=True,
            ordering={"indicators": False},
            columnDefs=[
                {
                    "className": "dt-center",
                    "target": "_all",
                }
            ],
        )

    def sample(self, n=None, frac=None, seed=None, ignore_index=True):
        """
        Create a stratified sample of the dataset.

        Samples rows based on the distribution of the target attribute.

        Parameters
        ----------
        n : int, optional
            Number of items from each group to sample.
        frac : float, optional
            Fraction of items from each group to sample.
        seed : int, optional
            Random state for reproducibility.
        ignore_index : bool, default True
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        SampleDataset
            A new dataset object containing the sampled records.
        """
        return SampleDataset(
            self,
            self.df.groupby(
                self.target,
                group_keys=False,
            )[self.df.columns]
            .apply(
                lambda x: x.sample(
                    n=n,
                    frac=frac,
                    random_state=seed,
                    ignore_index=ignore_index,
                )
            )
            .reset_index(drop=True),
        )

    def _repr_html_(self):
        """
        Get the HTML representation of the dataset.

        Returns
        -------
        str
            HTML string for Jupyter display.
        """
        return self.df._repr_html_()


class SampleDataset(Dataset):
    """
    A specialized Dataset class for representing a sample of records.

    This class is typically instantiated via the `sample` method of the
    base `Dataset`. It maintains a reference to the original dataset's
    metadata while holding only a sample of the data.

    Parameters
    ----------
    dataset : Dataset
        The parent Dataset instance from which this sample was derived.
    df : DataFrame
        The sampled data records.

    Attributes
    ----------
    all_sample_datasets : list
        A class-level list containing all instantiated SampleDataset objects.
    """

    all_sample_datasets = []

    def __init__(self, dataset: Dataset, df: DataFrame):
        """
        Initialize the SampleDataset.

        Parameters
        ----------
        dataset : Dataset
            The source dataset.
        df : DataFrame
            The filtered/sampled records to be stored.
        """
        super().__init__(dataset.name)
        self.__df = ITableDF(
            df,
            table_name=f"SAMPLE {self.name.upper()}",
        )
        SampleDataset.all_sample_datasets.append(self)

    def reload_df(self):
        """
        Unlike the base Dataset class, a SampleDataset cannot easily "reload"
        from disk because it is a result of an in-memory sampling operation.
        Thus, consider re-running the logic from `Dataset.sample` instead.

        Returns
        -------
        ITableDF
            The currently held sampled data.
        """
        print(
            f"""
        This dataset was sampled from '{self.name}'. 
        This function may not perform correctly! 
        Consider sampling '{self.name}' again. 
        """
        )
        return self.__df

    @property
    def df(self):
        """
        Access the sampled data.

        Returns
        -------
        ITableDF
            The sampled data.
        """
        if self.__df is None:
            self.reload_df()
        return self.__df

    def __str__(self):
        """
        Return the string representation of the sample.

        Returns
        -------
        str
            The word 'SAMPLE' followed by the uppercase dataset name.
        """
        return "SAMPLE " + self.name.upper()
