# +
import json
from functools import cached_property
from os import path as os_path

from pandas import read_csv


# -

class HierarchiesDict(dict):
    def __init__(
        self,
        hierarchies_dir: str,
        num_of_attributes: int,
        qids: list,
        qids_idx: list,
    ):
        self.hierarchies_dir = hierarchies_dir
        self.__qids = []
        for pos, qid in enumerate(qids):
            self.__qids.extend([None] * (qids_idx[pos] - len(self.__qids)))
            self.__qids.append(qid)
        self.__qids.extend([None] * (num_of_attributes - len(self.__qids)))

    def __getitem__(self, key):
        if isinstance(key, int):
            _key = self.__qids[key]
            if _key is None:
                raise AttributeError(f"Cannot find hierarchy for attribute at index '{key}'")
        else:
            _key = key
        if _key not in self.keys():
            try:
                with open(f"{self.hierarchies_dir}/{_key}.json") as f:
                    super().__setitem__(_key, json.load(f))
            except:
                raise FileNotFoundError(f'Cannot find the hierarchy "{_key}".')
        return super().__getitem__(_key)


class Dataset:
    all_datasets = []

    def __init__(self, name: str):
        self.name = name
        self.__df = None
        self.__hierarchies = None
        self.__props = None
        Dataset.all_datasets.append(self)

    def __str__(self):
        return self.name

    @cached_property
    def path(self):
        return f"{os_path.dirname(__file__)}/{self.name}"

    @cached_property
    def props(self):
        if self.__props is None:
            with open(f"{self.path}/props.json") as f:
                _props = json.load(f)
            self.__props = _props
        return self.__props

    @cached_property
    def qids(self):
        return [self.df.columns[x] for x in self.props["qi_index"]]

    @cached_property
    def qids_idx(self):
        return self.props["qi_index"]

    @cached_property
    def is_categorical(self):
        return self.props["is_category"]

    @property
    def df(self):
        if self.__df is None:
            self.reload_df()
        return self.__df

    @property
    def hierarchies(self):
        if self.__hierarchies is None:
            self.__hierarchies = HierarchiesDict(
                f"{self.path}/hierarchies",
                len(list(self.df)),
                self.qids,
                self.qids_idx,
            )
        return self.__hierarchies

    def reload_df(self):
        self.__df = read_csv(f"{self.path}/{self.name}.csv")

    def _repr_html_(self):
        return self.df._repr_html_()
