import json
from functools import cached_property
from typing import Literal

import pandas as pd
from ipywidgets import Dropdown, Label, Stack, VBox, jslink

from k_anonymization.utils.data_table import get_ITable_widget

from .frame import ITableDF


class Hierarchy:
    """
    Attribute's generalization hierarchy.

    This class stores the generalization mapping of a QID attribute's
    generalization hierarchy and provides utility functions for acquiring
    necessary properties of the hierarchy and its nodes.

    Parameters
    ----------
    name : str
        The identifier for the hierarchy (usually the column name).
    hierarchy_df : pd.DataFrame
        A DataFrame where column 0 is the raw data and subsequent
        columns are progressively more generalized levels.
    """

    def __init__(self, name: str, hierarchy_df: pd.DataFrame):
        """
        Initialize from a pre-computed hierarchy mapping DataFrame.

        Parameters
        ----------
        name : str
            Hierarchy name.
        hierarchy_df : pd.DataFrame
            The pre-computed hierarchy mapping DataFrame.
        """
        hierarchy_df.sort_values(
            hierarchy_df.columns[::-1].to_list(),
            ignore_index=True,
            inplace=True,
        )
        self.__hierarchy_df = hierarchy_df
        self.__name = name

    @classmethod
    def from_csv(cls, name: str, path: str, sep: str = ","):
        """
        Initialize from a CSV file.

        The CSV should have no header, where the first column is the raw
        data and each following column is a higher level of generalization.

        Parameters
        ----------
        name : str
            Hierarchy name.
        path : str
            The path of the csv file.
        sep : str, default ','
            Separator string (delimiter) of the csv file.
        """
        return cls(name, pd.read_csv(path, sep=sep, header=None))

    @classmethod
    def from_json(cls, name: str, org_column: pd.DataFrame, json_path: str):
        """
        Initialize from a JSON configuration file.

        Supports two types of definition for generalization:

        1. ``lambda``: Apply lambda functions to derive the next
           generalization level based on the current value.

        2. ``tree``: Explicitly map a list of ``original`` values to a
           ``generalized`` value.

        Parameters
        ----------
        name : str
            Hierarchy name.
        org_column : pd.DataFrame
            The column from the original dataset to use as Level 0.
        json_path : str
            Path to the JSON configuration file.
        """
        hierarchy_df = pd.DataFrame({0: org_column.unique()})
        try:
            with open(json_path) as f:
                props = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'Cannot find hierarchy file at "{json_path}".')
        if "lambda" in props:
            for level in range(len(props["lambda"])):
                f = eval(props["lambda"][level])
                hierarchy_df[level + 1] = hierarchy_df[level].apply(f)
        else:
            for level in range(len(props["tree"])):
                _hi = hierarchy_df[[level]].copy()
                if props["tree"][level]["is_suppressed"]:
                    hierarchy_df[level + 1] = "*"
                else:
                    for v in props["tree"][level]["values"]:
                        _hi[_hi.isin(v["original"]).any(axis=1)] = v["generalized"]
                    hierarchy_df[level + 1] = _hi[level].copy()

        return cls(name, hierarchy_df)

    @property
    def name(self):
        """
        Name of this hierarchy.

        Returns
        -------
        str
        """
        return self.__name

    @property
    def hierarchy_df(self):
        """
        The underlying hierarchy mapping DataFrame.

        Returns
        -------
        ITableDF
        """
        return ITableDF(self.__hierarchy_df.copy())

    @cached_property
    def height(self):
        """
        Height of this hierarchy.

        Returns
        -------
        int
        """
        return self.__hierarchy_df.shape[1] - 1

    @cached_property
    def leaves(self):
        """
        List of leaves (values at level 0).

        Returns
        -------
        list
        """
        return self.__hierarchy_df[0].to_list()

    def contains(self, node_value: any):
        """
        Check whether a node exists anywhere in the hierarchy.

        Parameters
        ----------
        node_value : any
            The value of the node to inspect.

        Returns
        -------
        bool
        """
        if node_value in self.leaves or node_value == "*":
            return True

        return True if self.__hierarchy_df.eq(node_value).any().sum() != 0 else False

    def get_leaves_under_node(self, node_value: any):
        """
        Get all leaves (values at level 0) under the given node.

        Parameters
        ----------
        node_value : any
            The value of the node to inspect.

        Returns
        -------
        list
            A list of leaves.
        """
        if node_value in self.leaves:
            return []

        if node_value == "*":
            return self.leaves

        search_df = self.__hierarchy_df[self.__hierarchy_df.eq(node_value).any(axis=1)]
        if search_df.shape[0] == 0:
            raise ValueError(
                f'Node "{node_value}" (type: {type(node_value)}) not found in the hierarchy.'
            )
        return search_df[0].to_list()

    def get_height_of_node(self, node_value: any):
        """
        Get the generalization level for a specific node.

        Parameters
        ----------
        node_value : any
            The value of the node to inspect.

        Returns
        -------
        int
            Generalization level of the input node.
        """
        if node_value in self.leaves:
            return 0

        if node_value == "*":
            return self.height

        search_df = self.__hierarchy_df.columns[
            self.__hierarchy_df.eq(node_value).any()
        ]
        if search_df.size == 0:
            raise ValueError(
                f'Node "{node_value}" (type: {type(node_value)}) not found in the hierarchy.'
            )
        return search_df.item()

    def get_lowest_common_ancestor(
        self,
        node_values: list,
        get_type: Literal["value", "height"] = "value",
    ):
        """
        Find the lowest common ancestor (LCA) of the given nodes.

        This is used to find the lowest-level generalized value that can
        hide a group of different values.

        Parameters
        ----------
        node_values : list
            The list of values to inspect.
        get_type : {'value', 'height'}, default 'value'
            Whether to return the LCA value or its generalization level.

        Returns
        -------
        str or int
            The LCA value or its height (generalization level).
        """
        assert len(node_values) > 1, "node_values must contains at least 2 values."

        for node_value in node_values:
            if not self.contains(node_value):
                raise ValueError(
                    f'Node "{node_value}" (type: {type(node_value)}) not found in the hierarchy.'
                )

        search_df = self.__hierarchy_df[
            self.__hierarchy_df.isin(node_values).any(axis=1)
        ]

        for level in range(self.height):
            _unique = search_df[level].unique()
            if _unique.size == 1:
                if get_type == "value":
                    return _unique.item()
                return level

        if get_type == "value":
            return "*"
        return self.height


class HierarchiesDict(dict):
    """
    Dictionary-like manager for attributes' generalization hierarchies.

    This class maps dataset column indices or attribute names to their
    respective hierarchy definitions. Hierarchy configurations are
    lazy-loaded of from JSON files stored in the dataset's directory.

    Parameters
    ----------
    hierarchies_dir : str
        The path to the directory containing JSON files for the hierarchies.
    df : pd.DataFrame
        The original data.
    qids : list
        A list of names of the QID attributes.
    qids_idx : list
        The column indices of the QID attributes.

    Attributes
    ----------
    hierarchies_dir : str
        Directory path where JSON hierarchy files are located.
    """

    def __init__(
        self,
        hierarchies_dir: str,
        df: pd.DataFrame,
        qids: list,
        qids_idx: list,
    ):
        """
        Initialize the HierarchiesDict.

        Constructs an internal mapping of attribute names to their specific
        column indices to allow for flexible lookup.

        Parameters
        ----------
        hierarchies_dir : str
            The path to the directory containing JSON files for the hierarchies.
        df : pd.DataFrame
            The original data.
        qids : list
            A list of names of the QID attributes.
        qids_idx : list
            The column indices of the QID attributes.
        """
        self.hierarchies_dir = hierarchies_dir
        self.__df = df
        self.__qids = []
        for pos, qid in enumerate(qids):
            self.__qids.extend([None] * (qids_idx[pos] - len(self.__qids)))
            self.__qids.append(qid)
        self.__qids.extend([None] * (df.columns.size - len(self.__qids)))

    def __getitem__(self, key):
        """
        Retrieve a Hierarchy by index or attribute name.

        If the hierarchy is not already loaded in memory, this method
        attempts to find and load the corresponding JSON file from the
        `hierarchies_dir`.

        Parameters
        ----------
        key : int or str
            The column index (int) or attribute name (str) to look up.

        Returns
        -------
        Hierarchy
            The hierarchy object associated with the specified attribute.

        Raises
        ------
        AttributeError
            If the key is an index with no assigned QID or if the
            attribute name is not found in the QID list.
        """
        if isinstance(key, int):
            _key = self.__qids[key]
            if _key is None:
                raise AttributeError(
                    f"Cannot find hierarchy for attribute at index '{key}'"
                )
        else:
            _key = key
            if _key is None or _key not in self.__qids:
                raise AttributeError(f"Cannot find attribute '{_key}'")
        if _key not in self.keys():
            super().__setitem__(
                _key,
                Hierarchy.from_json(
                    _key,
                    self.__df[_key],
                    f"{self.hierarchies_dir}/{_key}.json",
                ),
            )
        return super().__getitem__(_key)

    @cached_property
    def all_hierarchies_df(self):
        """
        Interactive widget to display all loaded hierarchies on IPython.

        Generates a UI component for IPython consisting of a dropdown menu
        and a display area showing the generalization hierarchy of each QID
        attribute.

        Returns
        -------
        ipywidgets.VBox
        """
        qids = [x for x in self.__qids if x is not None]
        dropdown = Dropdown(options=qids, value=qids[0])
        children = [
            get_ITable_widget(self.__getitem__(qid).hierarchy_df) for qid in qids
        ]
        stack = Stack(children, selected_index=0)
        dropdown = Dropdown(options=qids)
        jslink((dropdown, "index"), (stack, "selected_index"))
        return VBox([Label("Generalization Trees:"), dropdown, stack])
