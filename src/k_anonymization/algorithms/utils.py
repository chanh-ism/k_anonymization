from numpy import ndarray, vectorize
from pandas.core.frame import DataFrame


def generalize_df(df: DataFrame, hierarchy: dict, level: int):
    is_suppressed = False

    np_data, is_suppressed = generalize(
        df.to_numpy(copy=True), hierarchy, list(df).index(hierarchy["name"]), level
    )

    return DataFrame(np_data, columns=list(df)), is_suppressed


def generalize(data: ndarray, hierarchy: dict, qid_idx: int, level: int):
    is_suppressed = False
    f = None

    if "lambda" in list(hierarchy):
        f = vectorize(eval(hierarchy["lambda"][level]))
        if level == len(hierarchy["lambda"]) - 1:
            is_suppressed = True
    else:
        is_suppressed = hierarchy["tree"][level]["is_suppressed"]
        if is_suppressed:
            f = vectorize(lambda x: "*")
        else:
            generalized_values = hierarchy["tree"][level]["values"]

            def find_generalized_value(x):
                for generalized_value in generalized_values:
                    if x in generalized_value["original"]:
                        return generalized_value["generalized"]

            f = vectorize(lambda x: find_generalized_value(x))

    data[:, qid_idx] = f(data[:, qid_idx])

    return data, is_suppressed


def generalize_column(values: list, hierarchy: dict, level: int):
    if "lambda" in list(hierarchy):
        return generalize_column_lambda(values, hierarchy, level)
    return generalize_column_tree(values, hierarchy, level)


def generalize_column_tree(values: list, hierarchy: dict, level: int):
    assert "tree" in list(
        hierarchy
    ), f"Taxonomy tree of '{hierarchy["name"]}' is not defined."
    assert level < len(
        hierarchy["tree"]
    ), f"Input level {level} exceeds the highest generalization level of '{hierarchy["name"]}'."

    is_suppressed = hierarchy["tree"][level]["is_suppressed"]
    if is_suppressed:
        return list(map(lambda x: "*", values))
    else:
        generalized_values = hierarchy["tree"][level]["values"]

        def find_generalized_value(x):
            for generalized_value in generalized_values:
                if x in generalized_value["original"]:
                    return generalized_value["generalized"]
            return x

        return list(map(find_generalized_value, values))


def generalize_column_lambda(values: list, hierarchy: dict, level: int):
    assert "lambda" in list(
        hierarchy
    ), f"Lambda functions for generalizing '{hierarchy["name"]}' are not defined."
    assert level < len(
        hierarchy["lambda"]
    ), f"Input level {level} exceeds the highest generalization level of '{hierarchy["name"]}'."

    f = eval(hierarchy["lambda"][level])

    return list(map(f, values))
