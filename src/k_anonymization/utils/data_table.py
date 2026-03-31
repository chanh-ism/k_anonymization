"""
Interactive DataFrame display on IPython using ``itables``.
"""

from IPython.display import HTML, display
from itables import JavascriptFunction, init_notebook_mode
from itables import show as itables_show
from itables import widget
from pandas.core.frame import DataFrame

css = """
.dt-scroll {
  margin: 0 0 !important;
}
.dtsp-panesContainer {
  width: 100% !important;
}
.dt-scroll-body {
  height: auto !important;
  border-bottom: none !important;
}

.dt-length label {
  padding-left: 4px;
}

div.dtsp-searchPane div.dtsp-topRow {
  border: 0.5px solid lightgrey !important;
}

thead:has(.no-header) {
  display: none;
}
"""

__init = False


def show(
    data: DataFrame,
    table_name: str = None,
    search_columns_per_row: int = 4,
    max_bytes: str = "64KB",
):
    """
    Interactive DataFrame display on IPython using ``itables``.

    Parameters
    ----------
    data : pandas.DataFrame
        The Pandas DataFrame to be displayed.
    table_name : str, optional
        An optional title to display above the table.
        If provided, the title is also used as an HTML id for CSS.
    search_columns_per_row : int, default 4
        Determines the grid layout of the SearchPanes (filter boxes).
    max_bytes : str, default "64KB"
        The maximum data size threshold for ``itables`` to render.
        If data size is larger than ``max_bytes``, it will be downsampled.
        To remove the size threshold, set ``max_bytes`` to ``0``.
        Note that this may cause a performance issue if data is large.

    Notes
    -----
    On the first call, this function performs a global initialization:

    1. Injects customized layout CSS.

    2. Initializes ``itables`` notebook mode with ``connected=True``.

    The resulting dataframe display includes:

    * **Pagination**: Display data in pages of maximum 10 records.

    * **Filters**: High-level filters for attributes values.

    * **Column Controls**: Individual dropdown searches per column.
    """
    global __init
    if not __init:
        display(HTML(f"<style>{css}</style>" ""))
        init_notebook_mode(
            # all_interactive=True,
            connected=True,
        )
        __init = True
    _layout = {
        "topStart": None,
        "topEnd": None,
        "bottomStart": None,
        "bottomEnd": None,
        "bottom1": ["paging"],
        "bottom2": ["info"],
    }
    if table_name:
        _table_name = table_name.replace(" ", "_").lower()
        _layout["top1"] = JavascriptFunction(
            f"""
            function () {'{'}
                let toolbar = document.createElement('div');
                toolbar.innerHTML = '<h2 align="center" style="margin-top: 0">{table_name}</h2>';
     
                return toolbar;
            {'}'}
            """
        )
    else:
        _table_name = "custom-itables"

    _layout["top2"] = {
        "id": f"{_table_name}-buttons",
        "features": [
            {
                "buttons": [
                    {
                        "extend": "searchPanes",
                        "config": {
                            "threshold": 1,
                            "layout": f"columns-{search_columns_per_row}",
                            "cascadePanes": True,
                            "initCollapsed": True,
                            "dtOpts": {"order": [[1, "desc"], [0, "asc"]]},
                        },
                    },
                    {
                        # "extend": "collection",
                        "text": "Reset All",
                        "action": JavascriptFunction(
                            """
                            function (e, dt, node, config) {
                                dt.columns().ccSearchClear()
                                dt.searchPanes.clearSelections()
                                dt.draw()
                            }
                            """
                        ),
                        "split": [
                            {"extend": "searchPanesClear", "text": "Reset Filters"},
                            {"extend": "ccSearchClear", "text": "Reset Search"},
                        ],
                    },
                ]
            },
            "pageLength",
        ],
    }

    table_css = f"""
        #{_table_name}-buttons .dt-button-collection.dtb-collection-closeable {"{"}
            padding: 0.5rem 0.5rem 1rem 0.5rem;
            left: 0 !important;
            overflow-y: auto !important;
            max-height: -moz-available;
            max-height: -webkit-fill-available;
            max-height: fill-available;
        {"}"}
        #{_table_name}-buttons div.dtsp-searchPanes {"{"}
            width: 100% !important;
            column-gap: 1% !important;
            justify-content: flex-start !important;
        {"}"}
    """
    display(HTML(f"<style>{table_css}</style>" ""))
    itables_show(
        data,
        maxBytes=max_bytes,
        layout=_layout,
        language={
            "searchPanes": {
                "collapse": {0: "Filters", "_": "Filters (%d)"},
            }
        },
        columnControl=["searchDropdown"],
        ordering={"indicators": False},
    )


def get_ITable_widget(
    data: DataFrame,
    table_name: str = None,
    search_columns_per_row: int = 4,
    max_bytes: str = "64KB",
):
    """
    ``ipywidgets``-compatible ``itables`` DataFrame display on IPython.

    Parameters
    ----------
    data : pandas.DataFrame
        The Pandas DataFrame to be displayed.
    table_name : str, optional
        An optional title to display above the table.
        If provided, the title is also used as an HTML id for CSS.
    search_columns_per_row : int, default 4
        Determines the grid layout of the SearchPanes (filter boxes).
    max_bytes : str, default "64KB"
        The maximum data size threshold for ``itables`` to render.
        If data size is larger than ``max_bytes``, it will be downsampled.
        To remove the size threshold, set ``max_bytes`` to ``0``.
        Note that this may cause a performance issue if data is large.

    Notes
    -----
    On the first call, this function performs a global initialization:

    1. Injects customized layout CSS.

    2. Initializes ``itables`` notebook mode with ``connected=True``.

    The resulting dataframe display includes:

    * **Pagination**: Display data in pages of maximum 10 records.

    * **Filters**: High-level filters for attributes values.

    * **Column Controls**: Individual dropdown searches per column.
    """
    global __init
    if not __init:
        display(HTML(f"<style>{css}</style>" ""))
        init_notebook_mode(
            # all_interactive=True,
            connected=True,
        )
        __init = True
    _layout = {
        "topStart": None,
        "topEnd": None,
        "bottomStart": None,
        "bottomEnd": None,
        "bottom1": ["paging"],
        "bottom2": ["info"],
    }
    if table_name:
        _table_name = table_name.replace(" ", "_").lower()
        _layout["top1"] = JavascriptFunction(
            f"""
            function () {'{'}
                let toolbar = document.createElement('div');
                toolbar.innerHTML = '<h2 align="center" style="margin-top: 0">{table_name}</h2>';
     
                return toolbar;
            {'}'}
            """
        )
    else:
        _table_name = "custom-itables"

    _layout["top2"] = {
        "id": f"{_table_name}-buttons",
        "features": [
            {
                "buttons": [
                    {
                        "extend": "searchPanes",
                        "config": {
                            "threshold": 1,
                            "layout": f"columns-{search_columns_per_row}",
                            "cascadePanes": True,
                            "initCollapsed": True,
                            "dtOpts": {"order": [[1, "desc"], [0, "asc"]]},
                        },
                    },
                    {
                        # "extend": "collection",
                        "text": "Reset All",
                        "action": JavascriptFunction(
                            """
                            function (e, dt, node, config) {
                                dt.columns().ccSearchClear()
                                dt.searchPanes.clearSelections()
                                dt.draw()
                            }
                            """
                        ),
                        "split": [
                            {"extend": "searchPanesClear", "text": "Reset Filters"},
                            {"extend": "ccSearchClear", "text": "Reset Search"},
                        ],
                    },
                ]
            },
            "pageLength",
        ],
    }

    table_css = f"""
        #{_table_name}-buttons .dt-button-collection.dtb-collection-closeable {"{"}
            padding: 0.5rem 0.5rem 1rem 0.5rem;
            left: 0 !important;
            overflow-y: auto !important;
            max-height: -moz-available;
            max-height: -webkit-fill-available;
            max-height: fill-available;
        {"}"}
        #{_table_name}-buttons div.dtsp-searchPanes {"{"}
            width: 100% !important;
            column-gap: 1% !important;
            justify-content: flex-start !important;
        {"}"}
    """
    display(HTML(f"<style>{table_css}</style>" ""))
    return widget.ITable(
        data,
        maxBytes=max_bytes,
        layout=_layout,
        language={
            "searchPanes": {
                "collapse": {0: "Filters", "_": "Filters (%d)"},
            }
        },
        columnControl=["searchDropdown"],
        ordering={"indicators": False},
    )
