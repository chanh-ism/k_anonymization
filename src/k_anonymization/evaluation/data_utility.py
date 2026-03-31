"""
Data utility metrics.
"""

from numpy import ndarray
from pandas import DataFrame

from k_anonymization.core import HierarchiesDict, Hierarchy

from .anonymity import get_equivalence_classes


class Discernibility:
    r"""
    Discernibility Metric (DM).

    The Discernibility Metric measures the degree of ambiguity of the data.
    It assigns a penalty to each record based on the size of the
    equivalence class it belongs to. Smaller equivalence classes sizes
    result in a lower (better) score, and suppressed records are penalized
    based on the total size of the data.

    .. math::
        DM = \sum^{all\_EQs} |EQ|^2 + |S| * |D|

    where :math:`|EQ|` is the size of an equivalence class, :math:`|S|` is
    the number of suppressed records, and :math:`|D|` is the size of data.

    """

    @staticmethod
    def calculate(
        data: DataFrame | ndarray,
        qids_idx: list,
        suppression_counts: int = 0,
    ):
        """
        Calculate the discernibility from the data.

        Parameters
        ----------
        data : DataFrame or ndarray
            The data to inspect.
        qids_idx : list
            The column indices of the QID attributes.
        suppression_counts : int, default 0
            The number of suppressed records.

        Returns
        -------
        float
            The calculated discernibility.
        """
        equivalence_classes = get_equivalence_classes(data, qids_idx)

        return sum(
            [x["count"] ** 2 for x in equivalence_classes]
        ) + suppression_counts * (data.shape[0] + suppression_counts)

    @staticmethod
    def calculate_from_equivalence_classes(
        equivalence_classes: list, suppression_counts: int = 0
    ):
        """
        Calculate the discernibility from equivalence classes.

        Parameters
        ----------
        equivalence_classes : list[{qid, count}]
            A list of dictionaries, where each dictionary contains a
            'count' key representing the size of an equivalence class.
        suppression_counts : int, default 0
            The number of suppressed records.

        Returns
        -------
        float
            The calculated discernibility.

        See Also
        --------
        k_anonymization.evaluation.anonymity.get_equivalence_classes
            Get all equivalence classes.
        """
        org_data_size = 0
        if suppression_counts > 0:
            org_data_size = suppression_counts + sum(
                [x["count"] for x in equivalence_classes]
            )

        return (
            sum([x["count"] ** 2 for x in equivalence_classes])
            + suppression_counts * org_data_size
        )

    @staticmethod
    def calculate_best_effort(org_data: DataFrame, k: int = 1):
        r"""
        Calculate the best-effort discernibility based on `k`.

        When data size (:math:`|D|`) is divisible by k, the best
        discernibility (DM) is equal to :math:`\frac{|D|}{k}*k^2`.

        Otherwise, let :math:`R = |D|\:mod\:k` be the number of
        remainder records. The best DM happens when each remainder record
        is grouped into one different equivalence class (EQ). This results
        in :math:`int(\frac{|D|}{k}) - R` EQs of size :math:`k`,
        and :math:`R` EQs of size :math:`k + 1`.


        Parameters
        ----------
        data : DataFrame or ndarray
            The data to inspect.
        k : int, default 1
            The privacy parameter `k`.

        Returns
        -------
        float
            The calculated best-effort discernibility.
        """
        _size = org_data.shape[0]
        _optimal_num_of_eqs = int(_size / k)
        _remainder_records = _size % k

        _dm_eq_size_k = (_optimal_num_of_eqs - _remainder_records) * k**2
        _dm_eq_size_k_plus_1 = _remainder_records * (k + 1) ** 2

        return _dm_eq_size_k + _dm_eq_size_k_plus_1


class CAVG:
    r"""
    (Normalized) Average Equivalence Class Size (:math:`C_{AVG}`).

    :math:`C_{AVG}` estimates the trade-off between information loss and
    privacy protection based on the average size of the equivalence classes.
    A value closer to 1 indicates a more balanced trade-off.
    :math:`C_{AVG}` believes that a minimal-loss `k`-anonymization algorithm
    is one that results in equivalence classes all of size `k`.
    In other words, an equivalence class of size > `k` express an
    over-anonymization that led to unnecessary information loss.

    .. math::
        C_{AVG} = \frac{|D|}{|EQs| * k}

    where :math:`|D|` is the size of data and :math:`|EQs|` is the number of
    equivalence classes.
    """

    @staticmethod
    def calculate(
        data: DataFrame | ndarray,
        qids_idx: list,
        k: int,
    ):
        """
        Calculate CAVG score from the data.

        Parameters
        ----------
        data : DataFrame or ndarray
            The data to inspect.
        qids_idx : list
            The column indices of the QID attributes.
        k : int
            The privacy parameter `k`.

        Returns
        -------
        float
            The calculated CAVG.
        """

        equivalence_classes = get_equivalence_classes(data, qids_idx)
        return float(data.shape[0] / (len(equivalence_classes) * k))

    @staticmethod
    def calculate_from_equivalence_classes(
        equivalence_classes: list,
        k: int,
    ):
        """
        Calculate CAVG from equivalence classes.

        Parameters
        ----------
        equivalence_classes : list[{qid, count}]
            A list of dictionaries, where each dictionary contains a
            'count' key representing the size of an equivalence class.
        k : int
            The privacy parameter `k`.

        Returns
        -------
        float
            The calculated CAVG.

        See Also
        --------
        k_anonymization.evaluation.anonymity.get_equivalence_classes
            Get all equivalence classes.
        """

        org_data_size = sum([x["count"] for x in equivalence_classes])
        return float(org_data_size / (len(equivalence_classes) * k))

    @staticmethod
    def calculate_best_effort(org_data: DataFrame, k: int = 1):
        r"""
        Calculate the best-effort CAVG.

        The best CAVG happens when data records are evenly distributed into
        :math:`int(\frac{|D|}{k})` equivalence classes, i.e., when the
        data has exactly :math:`int(\frac{|D|}{k})` equivalence classes.

        .. math::
            C_{AVG}\_best = \frac{|D|}{int(\frac{|D|}{k}) * k}

        where :math:`|D|` is the size of data.

        Parameters
        ----------
        org_data : DataFrame
            The original data.
        k : int, default 1
            The privacy parameter `k`.

        Returns
        -------
        float
            The calculated best-effort CAVG.
        """
        _size = org_data.shape[0]
        _optimal_num_of_eqs = int(_size / k)

        return float(_size / (_optimal_num_of_eqs * k))


class NCP:
    r"""
    Normalized Certainty Penalty (NCP).

    NCP applies value-wise penalty on each value of QID attribute and
    normalizes it to [0, 1] by the data size :math:`|D|` and the number of
    QIDs :math:`|Q|`. A lower NCP indicates a lower information loss.


    .. math::
        NCP = \frac{1}{|D|} *
        \sum^{|D|}\frac{\sum P_{num}(v_{num}) + \sum P_{cat}(v_{cat})}{|Q|}

    where :math:`P_{num}(v_{num})` is the penalty for a value of a numerical
    QID attribute, :math:`P_{cat}(v_{cat})` is the penalty for a value of
    a categorical QID attribute. Depending on the anonymization method,
    :math:`P_{num}` and :math:`P_{cat}` are calculated diferrently.
    """

    @staticmethod
    def calculate_for_generalization(
        org_data: DataFrame,
        anon_data: DataFrame,
        hierarchies: HierarchiesDict,
        qids_idx: list,
        is_categorical: list,
    ):
        r"""
        Calculate NCP for generalization anonymization.

        When a numerical value is generalized to a (local) numerical range,
        it becomes ambiguous in such a range.
        Thus, :math:`P_{num} = \frac{local\_range}{global\_range}`.

        For categorical value, it becomes ambiguous among the leaves under
        the common ancestor for its equivalence class.
        Thus :math:`P_{cat} = \frac{leaves\_under\_common\_ancestor}{all\_leaves}`.

        Parameters
        ----------
        org_data : DataFrame
            The original data.
        anon_data : DataFrame
            The anonymized data.
        hierarchies : HierarchiesDict
            Hierarchy definitions for the QID attributes.
        qids_idx : list
            The column indices of the QID attributes.
        is_categorical : list
            A list of booleans indicating if a QID attribute is categorical.

        Returns
        -------
        float
            The NCP score.
        """
        equivalence_classes = get_equivalence_classes(anon_data, qids_idx)

        _all_penalties = 0.0
        _max_ranges = {}

        def get_penalty_cat(value, hierarchy: Hierarchy):
            return len(hierarchy.get_leaves_under_node(value)) / len(hierarchy.leaves)

        for qid in equivalence_classes:
            _penalty = 0.0
            _size = qid["count"]
            for pos, val in enumerate(qid["qid"]):
                if val == "*":
                    _penalty += 1
                elif is_categorical[pos]:
                    _penalty += get_penalty_cat(val, hierarchies[qids_idx[pos]])
                else:
                    if not isinstance(val, str):
                        # Assume that if a value is still in numerical form,
                        # it is not generalized...
                        continue
                    if "~" in val:
                        # If numerical value is generalized as a range
                        low, high = val.split("~")
                        if pos not in _max_ranges:
                            org_low = org_data.iloc[:, qids_idx[pos]].min()
                            org_high = org_data.iloc[:, qids_idx[pos]].max()
                            _max_ranges[pos] = org_high - org_low
                        _penalty += (float(high) - float(low)) / _max_ranges[pos]

                    # elif val.endswith("*"):
                    # TODO: If numerical value is generalized by removing tailing digit(s)
                    #       For example: Zip code (12345 -> 1234*)

                    else:
                        continue

            _all_penalties += _penalty * _size

        _all_penalties += (org_data.shape[0] - anon_data.shape[0]) * len(qids_idx)
        return _all_penalties / (len(qids_idx) * org_data.shape[0])

    @staticmethod
    def calculate_for_local_recoding_mean_mode(
        org_data: DataFrame,
        groups: list,
        qids_idx: list,
        is_categorical: list,
    ):
        r"""
        Calculate NCP for local recoding algorithm with mean-mode group anonymization.

        When a numerical value is generalized to a (local) numerical range,
        it becomes ambiguous in such a range.
        Thus, :math:`P_{num} = \frac{local\_range}{global\_range}`.

        For categorical value, if an original value is different from the
        mode of its equivalence class, the entire value is loss.
        Thus :math:`P_{cat}(v_{cat}) = 1\:if\:v_{cat} \neq mode, 0\:otherwise`.

        Parameters
        ----------
        org_data : DataFrame
            The original data.
        groups : list
            The anonymized data.
        qids_idx : list
            The column indices of the QID attributes.
        is_categorical : list
            A list of booleans indicating if a QID attribute is categorical.

        Returns
        -------
        float
            The NCP score.

        See Also
        --------
        k_anonymization.algorithms.local_recoding.GroupAnonymizationBuiltIn.MEAN_MODE
            Anonymize a group by mean and mode.
        """
        _all_penalties = 0.0
        _max_ranges = {}
        _sum_groups_sizes = 0

        for group in groups:
            _penalty = 0.0
            _size = len(group)
            _sum_groups_sizes += _size
            columns = list(zip(*group))
            for pos, idx in enumerate(qids_idx):
                values = columns[idx]
                if is_categorical[pos]:
                    mode = max(values, key=values.count)
                    _penalty += 1 - (values.count(mode) / _size)
                else:
                    if pos not in _max_ranges:
                        org_low = org_data.iloc[:, qids_idx[pos]].min()
                        org_high = org_data.iloc[:, qids_idx[pos]].max()
                        _max_ranges[pos] = org_high - org_low
                    _penalty += (max(values) - min(values)) / _max_ranges[pos]
            _all_penalties += _penalty * _size

        _all_penalties += (org_data.shape[0] - _sum_groups_sizes) * len(qids_idx)
        return _all_penalties / (len(qids_idx) * org_data.shape[0])

    @staticmethod
    def calculate_for_local_recoding_summarization(
        org_data: DataFrame,
        groups: list,
        qids_idx: list,
        is_categorical: list,
    ):
        r"""
        Calculate NCP for local recoding algorithm with summarization group anonymization.

        When a numerical value is generalized to a (local) numerical range,
        it becomes ambiguous in such a range.
        Thus, :math:`P_{num} = \frac{local\_range}{global\_range}`.

        For categorical value, it becomes ambiguous among the unique values
        of its QID attribute in its equivalence class, denoted as
        :math:`Q^{EQ}(v_{cat})`.
        Thus :math:`P_{cat}(v_{cat}) = \frac{1}{|Q^{EQ}(v_{cat}).unique|}`.

        Parameters
        ----------
        org_data : DataFrame
            The original data.
        groups : list
            The anonymized data.
        qids_idx : list
            The column indices of the QID attributes.
        is_categorical : list
            A list of booleans indicating if a QID attribute is categorical.

        Returns
        -------
        float
            The NCP score.

        See Also
        --------
        k_anonymization.algorithms.local_recoding.GroupAnonymizationBuiltIn.SUMMARIZATION
            Anonymize a group by creating a summary range or set.
        """
        _all_penalties = 0.0
        _max_ranges = {}
        _sum_groups_sizes = 0

        for group in groups:
            _penalty = 0.0
            _size = len(group)
            _sum_groups_sizes += _size
            columns = list(zip(*group))
            for pos, idx in enumerate(qids_idx):
                values = columns[idx]
                if is_categorical[pos]:
                    if pos not in _max_ranges:
                        _max_ranges[pos] = org_data.iloc[:, qids_idx[pos]].unique().size
                    _penalty += len(set(values)) / _max_ranges[pos]
                else:
                    if pos not in _max_ranges:
                        org_low = org_data.iloc[:, qids_idx[pos]].min()
                        org_high = org_data.iloc[:, qids_idx[pos]].max()
                        _max_ranges[pos] = org_high - org_low
                    _penalty += (max(values) - min(values)) / _max_ranges[pos]
            _all_penalties += _penalty * _size

        _all_penalties += (org_data.shape[0] - _sum_groups_sizes) * len(qids_idx)
        return _all_penalties / (len(qids_idx) * org_data.shape[0])
