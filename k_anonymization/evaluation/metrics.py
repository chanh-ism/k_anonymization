# +
from pandas.core.frame import DataFrame

from ..algorithms.type import Algorithm
from .utils import count_equivalent_qids


# -

def discernibility(
    anon_df: DataFrame,
    qids: list = [],
    suppressed_qids: int = 0,
    org_df_size: int = 0,
):
    num_of_equivalent_qids = count_equivalent_qids(anon_df, qids=qids)
    return sum([x**2 for x in num_of_equivalent_qids]) + suppressed_qids * org_df_size


def discernibility_from_algo(algo: Algorithm):
    return discernibility(
        algo.anon_data,
        algo.dataset.qids,
        (
            sum([x["count"] for x in algo.suppressed_qids])
            if len(algo.suppressed_qids) > 0
            else 0
        ),
        algo.org_data.shape[0],
    )
