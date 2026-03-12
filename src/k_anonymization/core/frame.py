from pandas import DataFrame

from k_anonymization.utils.data_table import show


class ITableDF(DataFrame):
    """
    A custom `pandas.DataFrame` class that integrates UI from `itables`.

    This class inherits from `pandas.DataFrame` and overrides its
    HTML representation to `itables`'s.

    Parameters
    ----------
    *args : tuple
        Positional arguments for the parent `pandas.DataFrame`.
    **kwargs : dict
        Keyword arguments for the parent `pandas.DataFrame`.
        Can optionally include a `table_name` key (str).

    Attributes
    ----------
    table_name : str or None
        The name assigned to the table, if provided during initialization.
    """

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
        """
        Override `pandas.DataFrame`'s HTML representation with `itables`'s.

        Returns
        -------
        str
            A hidden HTML italic tag `<i hidden></i>`.
        """
        show(self, self.table_name)
        return "<i hidden></i>"

    def show_whole_table(self):
        """
        Display the entirety of the table.

        Calls the external `show` function with `max_bytes=0` to ensure
        that display truncation or byte-size limits are bypassed.
        Note that this may cause a performance issue if data is large.
        """
        show(self, self.table_name, max_bytes=0)
