# +
import json
from functools import cached_property
from os import path as os_path

from graphviz import Digraph
from ipywidgets import Dropdown, Image, Label, Layout, Stack, VBox, jslink
from itables import show as itables_show
from pandas import DataFrame, read_csv

from k_anonymization.utils.data_table import show


# -

class DataFrameTable(DataFrame):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        table_name = None
        if "table_name" in kwargs:
            table_name = kwargs.pop("table_name")
        super().__init__(*args, **kwargs)
        self.table_name = table_name

    def _repr_html_(self):
        show(self, self.table_name)
        return "<i hidden></i>"

    def show_whole_table(self):
        show(self, self.table_name, max_bytes=0)


class HierarchiesDict(dict):
    def __init__(
        self,
        hierarchies_dir: str,
        num_of_attributes: int,
        qids: list,
        qids_idx: list,
        first_row: list,
    ):
        self.hierarchies_dir = hierarchies_dir
        self.__qids = []
        for pos, qid in enumerate(qids):
            self.__qids.extend([None] * (qids_idx[pos] - len(self.__qids)))
            self.__qids.append(qid)
        self.__qids.extend([None] * (num_of_attributes - len(self.__qids)))
        self.first_row = first_row

    def __getitem__(self, key):
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
            try:
                with open(f"{self.hierarchies_dir}/{_key}.json") as f:
                    super().__setitem__(
                        _key,
                        HierarchyDict(
                            self.first_row[self.__qids.index(_key)],
                            json.load(f),
                        ),
                    )
            except:
                raise FileNotFoundError(f'Cannot find hierarchy file of "{_key}".')
        return super().__getitem__(_key)

    @cached_property
    def generalization_trees(self):
        qids = [x for x in self.__qids if x is not None]
        children = [
            Image(
                value=self.__getitem__(qid).generalization_tree.pipe(format="png"),
                format="png",
                layout=Layout(max_width="unset"),
            )
            for qid in qids
        ]
        stack = Stack(children, selected_index=0)
        dropdown = Dropdown(options=qids)
        jslink((dropdown, "index"), (stack, "selected_index"))
        return VBox([Label("Generalization Trees:"), dropdown, stack])


class HierarchyDict(dict):
    def __init__(
        self,
        first_row,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.first_row = first_row
        if "tree" in self.keys():
            self.height = len(self.__getitem__("tree"))

    @cached_property
    def generalization_tree(self):
        if "tree" in self.keys():
            return self.__get_generalization_tree_from_tree()
        elif "lambda" in self.keys():
            return self.__get_generalization_tree_from_lambda()
        else:
            return "No generalization provided/found."

    def __get_generalization_tree_from_tree(self):
        tree = Digraph(
            edge_attr={"arrowhead": "none"},
            node_attr={"shape": "plaintext"},
        )
        name = self.__getitem__("name")
        if self.height == 1:
            tree.edge(f"(1)\n{name}", "All values")
            return tree
        for index, node in enumerate(self.__getitem__("tree")[:-1]):
            for values in node["values"]:
                tree.edges(
                    [
                        [
                            f'({index + 1})\n{values["generalized"]}',
                            f'{f"({index})\n" if index != 0 else ''}{v}',
                        ]
                        for v in values["original"]
                    ]
                )
            if index >= self.height - 2:
                tree.edges(
                    [
                        [
                            f"({index + 2})\n{name}",
                            f'({index + 1})\n{values["generalized"]}',
                        ]
                        for values in node["values"]
                    ]
                )
        return tree

    def __get_generalization_tree_from_lambda(self):
        tree = Digraph(graph_attr={"rankdir": "LR"})
        lambdas = self.__getitem__("lambda")
        nodes = [self.first_row]
        for f in lambdas:
            nodes.append(eval(f)(nodes[-1]))
        for i, v in enumerate(nodes[0:-1]):
            if i == 0:
                tree.edge(str(nodes[0]), str(nodes[1]), label="(1)")
            else:
                tree.edge(str(nodes[i]), str(nodes[i + 1]), label=f"({i+1})")
        return tree


class Dataset:
    all_datasets = []

    def __init__(self, name: str):
        self.name = name
        self.__df = None
        self.__hierarchies = None
        self.__props = None
        Dataset.all_datasets.append(self)

    def __str__(self):
        return self.name.upper()

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

    @cached_property
    def target(self):
        return self.props["target"]

    @cached_property
    def qids_categorial(self):
        return [qid for pos, qid in enumerate(self.qids) if self.is_categorical[pos]]

    @cached_property
    def qids_numerical(self):
        return [qid for pos, qid in enumerate(self.qids) if not self.is_categorical[pos]]

    @cached_property
    def qids_idx_categorial(self):
        return [idx for pos, idx in enumerate(self.qids_idx) if self.is_categorical[pos]]

    @cached_property
    def qids_idx_numerical(self):
        return [idx for pos, idx in enumerate(self.qids_idx) if not self.is_categorical[pos]]

    # TODO: Update other modules to use 
    # - qids_categorial
    # - qids_numerical
    # - qids_idx_categorial
    # - qids_idx_numerical

    @cached_property
    def info(self):
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
                self.df[0:1].values[0].tolist(),
            )
        return self.__hierarchies

    def reload_df(self):
        self.__df = DataFrameTable(
          read_csv(f"{self.path}/{self.name}.csv"), 
          table_name=self.name.upper(),
        )

    def describe(self):
        itables_show(
            DataFrame(
                [
                    self.name.upper(),
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

    def _repr_html_(self):
        return self.df._repr_html_()
