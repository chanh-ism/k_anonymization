import pandas as pd
from typing import List


def generalize(df: pd.DataFrame, hierarchy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generalize the DataFrame based on the provided hierarchy.

    df: Input DataFrame to be generalized.
    hierarchy_df: DataFrame containing the hierarchy mapping. Each row should be generalization target, and have col, from, to information.
    return: Generalized DataFrame.

    hierarchy_df should be like:
       column        child child_level         parent parent_level
    workclass  Federal-gov           0  In-government            1
    workclass    Local-gov           0  In-government            1
    workclass    State-gov           0  In-government            1
    ...
    """

    generalized_df = df.copy()

    # 各カラムについて、hierarchy_dfによる一般化
    for generalize_col in hierarchy_df["column"].unique():
        # 一般化規約の抽出
        mapping = hierarchy_df[hierarchy_df["column"] == generalize_col][
            ["child", "parent"]
        ]

        # 各列について一般化規則を反映（インデックスを維持）
        merged = (
            generalized_df[[generalize_col]]
            .reset_index()
            .merge(mapping, left_on=generalize_col, right_on="child", how="left")
            .set_index("index")
        )

        # インデックスを維持したまま一般化を適用
        generalized_df[generalize_col] = merged["parent"].where(
            merged["parent"].notna(), merged[generalize_col]
        )
    return generalized_df


def is_k_anonymous(
    df: pd.DataFrame, target_cols: List[str], k: int, debug: bool = False
) -> bool:
    """
    dfがtarget_colsにおいてk-匿名であるか確認する

    df: Input DataFrame
    target_cols: List of columns to check for k-anonymity.
    k: The value of k for k-anonymity.
    return: True if the DataFrame is k-anonymous, False otherwise.
    """

    # 各target_colsの組み合わせでグループ化し、サイズをカウント
    _df = df.copy()
    grouped = _df.groupby(target_cols, dropna=False)
    if debug:
        print(grouped.size())
    # 各グループのサイズがk以上であるか確認
    is_k_anonymous = all(size >= k for size in grouped.size())

    return is_k_anonymous
