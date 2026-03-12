import json
from functools import cached_property

from graphviz import Digraph
from ipywidgets import Dropdown, Image, Label, Layout, Stack, VBox, jslink


class Hierarchy(dict):
    """
    Attribute's generalization hierarchy.

    This class stores hierarchical structures used for data generalization,
    supporting both explicit tree definitions and functional lambda-based
    transformations. It also provides a visualization of these hierarchies
    via Graphviz Digraphs.

    Parameters
    ----------
    first_row : any
        The original data value of the first row used for visualizing
        generalization tree (for lambda-based available hierarchy).
    *args : tuple
        Positional arguments passed to the parent `dict` constructor.
    **kwargs : dict
        Keyword arguments passed to the parent `dict` constructor.

    Attributes
    ----------
    first_row : any
        The starting value for hierarchy transformations.
    height : int
        The number of levels in the hierarchy.
    """

    def __init__(
        self,
        first_row=None,
        *args,
        **kwargs,
    ):
        """
        Initialize the Hierarchy.

        Sets the first row and calculates the hierarchy height if a
        tree structure is provided in the input keys.

        Parameters
        ----------
        first_row : any
            The original data value of the first row.
        *args : tuple
            Arguments for `dict` initialization.
        **kwargs : dict
            Key-value pairs for `dict` initialization.
        """
        super().__init__(*args, **kwargs)
        self.first_row = first_row
        if "tree" in self.keys():
            self.height = len(self.__getitem__("tree"))

    @cached_property
    def generalization_tree(self):
        """
        Generate a visual representation of the generalization hierarchy.

        Depending on whether the dictionary contains a "tree" or "lambda"
        key, this property constructs a Graphviz Digraph.

        Returns
        -------
        graphviz.Digraph or str
            A Digraph object if a hierarchy is found, or a string message
            if no generalization logic is provided.
        """
        if "tree" in self.keys():
            return self.__get_generalization_tree_from_tree()
        elif "lambda" in self.keys():
            return self.__get_generalization_tree_from_lambda()
        else:
            return "No generalization provided/found."

    def __get_generalization_tree_from_tree(self):
        """
        Construct a Digraph from an explicit tree structure.

        Iterates through the "tree" list to map original values to
        their generalized counterparts across different levels.

        Returns
        -------
        graphviz.Digraph
            A visual tree showing the many-to-one mapping of values.
        """
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
                            f'{f"({index})\n" if index != 0 else ""}{v}',
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
        """
        Construct a Digraph from a sequence of lambda transformations.

        Evaluates the strings in the "lambda" key sequentially, applying
        each function to the result of the previous step starting
        from `first_row`.

        Returns
        -------
        graphviz.Digraph
            A visual linear path showing the transformation of a
            specific value through the hierarchy.
        """
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
    num_of_attributes : int
        The total number of attributes in the dataset.
    qids : list
        A list of names for the Quasi-Identifiers (QIDs).
    qids_idx : list
        A list of integer indices corresponding to the positions of the
        QIDs in the dataset.
    first_row : list
        A sample row of data used to initialize individual HierarchyDicts
        for lambda-based transformations.

    Attributes
    ----------
    hierarchies_dir : str
        Directory path where JSON hierarchy files are located.
    first_row : list
        Sample data used as the base for generalization levels.
    """

    def __init__(
        self,
        hierarchies_dir: str,
        num_of_attributes: int,
        qids: list,
        qids_idx: list,
        first_row: list,
    ):
        """
        Initialize the HierarchiesDict and map attributes to positions.

        Constructs an internal mapping of attribute names to their specific
        column indices to allow for flexible lookup.

        Parameters
        ----------
        hierarchies_dir : str
            Path to the hierarchy JSON files.
        num_of_attributes : int
            Total column count.
        qids : list
            Names of the Quasi-Identifiers.
        qids_idx : list
            Column indices of the Quasi-Identifiers.
        first_row : list
            Sample data values.
        """
        self.hierarchies_dir = hierarchies_dir
        self.__qids = []
        for pos, qid in enumerate(qids):
            self.__qids.extend([None] * (qids_idx[pos] - len(self.__qids)))
            self.__qids.append(qid)
        self.__qids.extend([None] * (num_of_attributes - len(self.__qids)))
        self.first_row = first_row

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
        FileNotFoundError
            If the JSON file for the requested attribute does not exist
            in the specified directory.
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
            try:
                with open(f"{self.hierarchies_dir}/{_key}.json") as f:
                    super().__setitem__(
                        _key,
                        Hierarchy(
                            self.first_row[self.__qids.index(_key)],
                            json.load(f),
                        ),
                    )
            except FileNotFoundError:
                raise FileNotFoundError(f'Cannot find hierarchy file of "{_key}".')
        return super().__getitem__(_key)

    @cached_property
    def generalization_trees(self):
        """
        Create an interactive widget to visualize all loaded hierarchies.

        Generates a UI component consisting of a dropdown menu and a
        display area that shows the Graphviz-rendered generalization
        tree for each attribute.

        Returns
        -------
        ipywidgets.VBox
            A container holding the dropdown and the stacked images
            of the generalization trees.
        """
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
