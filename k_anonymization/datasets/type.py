# +
import json
from os import path as os_path

from pandas import read_csv


# -

class HierarchiesDict(dict):
    def __init__(self, hierarchies_dir: str):
        self.hierarchies_dir = hierarchies_dir
    def __getitem__(self, key):
        if key not in self.keys():
            try:
                with open(f"{self.hierarchies_dir}/{key}.json") as f:
                    super().__setitem__(key, json.load(f))
            except:
                raise FileNotFoundError(f"Cannot find the hierarchy \"{key}\".")
        return super().__getitem__(key)


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

    @property
    def path(self):
        return f"{os_path.dirname(__file__)}/{self.name}"

    @property
    def props(self):
        if self.__props is None:
            with open(f"{self.path}/props.json") as f:
                _props = json.load(f)
            self.__props = _props
        return self.__props

    @property
    def qids(self):
        return [self.df.columns[x] for x in self.props["qi_index"]]

    @property
    def qids_idx(self):
        return self.props["qi_index"]

    @property
    def df(self):
        if self.__df is None:
            self.reload_df()
        return self.__df

    @property
    def hierarchies(self):
        if self.__hierarchies is None:
            self.__hierarchies = HierarchiesDict(f"{self.path}/hierarchies")
        return self.__hierarchies

    def reload_df(self):
        self.__df = read_csv(f"{self.path}/{self.name}.csv")

    def _repr_html_(self):
        return self.df.to_html()
