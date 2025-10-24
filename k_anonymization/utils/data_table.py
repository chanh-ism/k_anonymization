from IPython.display import HTML, display
from itables import init_notebook_mode, JavascriptFunction
from itables import show as itables_show
from pandas.core.frame import DataFrame

css = """
.jp-RenderedJSON .filter {
  display: none;
}
.dt-scroll {
  margin: 0 0 !important;
}
.dtsp-panesContainer {
  width: 100% !important;
}
.dt-scroll-body {
  height: auto !important;
}

.dt-length label {
  padding-left: 4px;
}

div.dtsp-panesContainer div.dtsp-searchPanes div.dtsp-searchPane {
  border: 0.5px solid lightgrey;
  border-radius: 4px;
}

"""
display(HTML(f"<style>{css}</style>" ""))

init_notebook_mode(all_interactive=True)


def show(
    data: DataFrame,
    table_name: str = None,
    search_columns_per_row: int = 4,
    max_bytes: str = "64KB",
):
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
            "pageLength",
            {
                "buttons": [
                    {
                        "extend": "searchPanes",
                        "config": {
                            "threshold": 1,
                            "columns": [i for i in range(1, len(data.keys()))],
                            "layout": f"columns-{search_columns_per_row}",
                            "cascadePanes": True,
                            "initCollapsed": True,
                            "dtOpts": {"order": [[1, "desc"], [0, "asc"]]},
                        },
                    },
                    "searchBuilder",
                ]
            },
        ],
    }

    table_css = f"""
        #{_table_name}-buttons .dt-button-collection {"{"}
            padding: 8px;
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
    )
