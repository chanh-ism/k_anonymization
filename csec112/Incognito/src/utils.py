import pandas as pd
import os

VERBOSE = False


def set_verbose(verbose: bool) -> None:
    """
    Set the global verbose flag for logging.
    """
    global VERBOSE
    VERBOSE = verbose


def vprint(*args, **kwargs) -> None:
    """
    Print messages only if verbose mode is enabled.
    """
    if VERBOSE:
        print(*args, **kwargs)


def read_dataset(dataset_name: str) -> pd.DataFrame:
    """
    データセットを Data ディレクトリから読み込む
    Dataset directory structure:
    ./Data/
        {dataset_name}/
            {dataset_name}.csv
            hierarchies/
                {column}.csv
                ...

    param: dataset_name: データセット名
    return: pd.DataFrame: 読み込んだデータセット
    """
    if dataset_name not in ["adult", "atus", "cup", "fars", "ihis", "ACS13_ma", "london_house_price"]:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Determine separator (ACS13_ma uses comma, others use semicolon)
    separator = "," if dataset_name in ["ACS13_ma", "london_house_price"] else ";"

    dataset_path = f"Data/{dataset_name}/{dataset_name}.csv"
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path, sep=separator)
    else:
        raise ValueError(f"Dataset file ({dataset_path}) does not exist.")

    return data


def read_hierarchy_official_csv(file_path: str, col_name: str) -> pd.DataFrame:
    """
    read hierarchy from official csv file
    param file_path: path to the hierarchy csv file
    return: hierarchy df
    """
    csv = pd.read_csv(file_path, sep=";", header=None)
    hierarchy_df = pd.DataFrame(
        columns=["child", "child_level", "parent", "parent_level"]
    )
    for child_col in range(csv.shape[1] - 1):
        for parent_col in range(child_col + 1, csv.shape[1]):
            csvf = csv.iloc[:, [child_col, parent_col]]
            csvf = csvf.drop_duplicates()
            append_df = pd.DataFrame(csvf.values, columns=["child", "parent"])
            append_df["child_level"] = child_col
            append_df["parent_level"] = parent_col
            hierarchy_df = pd.concat([hierarchy_df, append_df], ignore_index=True)

    # csvとdatatable上のcol名が違うものは置換する
    if col_name == "salary-class":
        col_name = "income"

    hierarchy_df["column"] = col_name
    return hierarchy_df


def read_hierarchies_by_col_names(col_names: list[str], hierarchies_dir: str) -> pd.DataFrame:
    """
    CSV階層ファイルから階層定義を読み込む

    param col_names: list of column names to read hierarchies
    param hierarchies_dir: directory containing hierarchy CSV files
    return: concatenated hierarchy df
    """
    if not os.path.exists(hierarchies_dir):
        raise ValueError(f"Hierarchies directory: {hierarchies_dir} does not exist.")

    hierarchies = []
    for col_name in col_names:
        hierarchy_path = os.path.join(hierarchies_dir, f"{col_name}.csv")
        if not os.path.exists(hierarchy_path):
            raise ValueError(f"Hierarchy file not found: {hierarchy_path}")
        hierarchies.append(read_hierarchy_official_csv(hierarchy_path, col_name))

    return pd.concat(hierarchies, ignore_index=True)


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN values.
    param df: Input DataFrame
    return: DataFrame with rows containing NaN in specified columns dropped
    """
    df = df.replace("?", pd.NA)  # '?'をNaNに置換
    df = df.dropna(axis=0, how="any")
    df = df.reset_index(
        drop=True
    )  # 欠番があるとgeneralizaのwhere句で存在しないインデックスを参照してNaNが生えてしまう

    return df
