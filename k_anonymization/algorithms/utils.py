from numpy import ndarray, vectorize
from pandas.core.frame import DataFrame


def generalize_df(df: DataFrame, hierarchy: dict, level: int):
    is_suppressed = False

    np_data, is_suppressed = generalize(
        df.values, hierarchy, list(df).index(hierarchy["name"]), level
    )

    return DataFrame(np_data, columns=list(df)), is_suppressed


def generalize(data: ndarray, hierarchy: dict, qid_idx: int, level: int):
    is_suppressed = False
    f = None

    if hierarchy["type"] == "lambda":
        f = vectorize(eval(hierarchy["lambda"][level]))
        if level == len(hierarchy["lambda"]) - 1:
            is_suppressed = True
    elif hierarchy["type"] == "list":
        is_suppressed = hierarchy["hierarchy"][level]["is_suppressed"]
        if is_suppressed:
            f = vectorize(lambda x: "*")
        else:
            generalized_values = hierarchy["hierarchy"][level]["values"]

            def find_generalized_value(x):
                for generalized_value in generalized_values:
                    if x in generalized_value["original"]:
                        return generalized_value["generalized"]

            f = vectorize(lambda x: find_generalized_value(x))

    data[:, qid_idx] = f(data[:, qid_idx])

    return data, is_suppressed
