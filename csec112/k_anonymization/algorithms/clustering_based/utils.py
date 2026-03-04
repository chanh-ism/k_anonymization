# +
from k_anonymization.datasets.type import Dataset

from ..utils import generalize_column_tree


# -

def get_max_ranges(dataset: Dataset):
    qids_idx = dataset.qids_idx
    is_cat = dataset.is_categorical
    hierarchies = dataset.hierarchies
    df = dataset.df
    columns = list(df)

    max_ranges = []
    for pos, idx in enumerate(qids_idx):
        max_ranges.extend([None] * (idx - len(max_ranges)))
        if is_cat[pos] == True:
            max_ranges.append(
                len(hierarchies[idx]["lambda"])
                if "lambda" in list(hierarchies[idx])
                else len(hierarchies[idx]["tree"])
            )
        else:
            max_ranges.append(df[columns[idx]].max() - df[columns[idx]].min())

    max_ranges.extend([None] * (len(columns) - len(max_ranges)))

    return max_ranges


def get_distance(r, record, qids_idx, is_cat, max_ranges, hierarchies):
    distances = []

    for pos, idx in enumerate(qids_idx):
        if is_cat[pos] == True:
            distances.append(
                get_categorical_distance(
                    [r[idx], record[idx]],
                    hierarchies[idx],
                    max_ranges[idx],
                )
            )
        else:
            distances.append(abs(r[idx] - record[idx]) / max_ranges[idx])

    return sum(distances)


def get_information_loss(record, cluster, qids_idx, is_cat, max_ranges, hierarchies):
    information_losses = []
    if record == None:
        size = len(cluster)
        columns = list(zip(*cluster))
    else:
        size = len(cluster) + 1
        columns = list(zip(*(cluster + [record])))

    for pos, idx in enumerate(qids_idx):
        if is_cat[pos] == True:
            information_losses.append(
                get_categorical_distance(
                    columns[idx], hierarchies[idx], max_ranges[idx]
                )
            )
        else:
            information_losses.append(
                (max(columns[idx]) - min(columns[idx])) / max_ranges[idx]
            )

    return size * sum(information_losses)


def get_categorical_distance(values, hierarchy, height):
    level = 0
    generalized_values = values[:]

    while len(set(generalized_values)) > 1:
        generalized_values = generalize_column_tree(
            generalized_values, hierarchy, level
        )
        level += 1

    return level / height


def summarize(values, is_cat):
    anon_value = None
    if is_cat == True:
        try:
            anon_value = " & ".join(set(values))
        except:
            anon_value = " & ".join([str(x) for x in set(values)])
    else:
        if len(set(values)) == 1:
            anon_value = f"{values[0]}"
        else:
            anon_value = f"{min(values)} ~ {max(values)}"
    return list(map(lambda x: anon_value, values))


def get_mean_mode(values, is_cat):
    anon_value = None
    if is_cat == True:
        anon_value = max(values, key=values.count)
    else:
        anon_value = sum(values) / len(values)
    return list(map(lambda x: anon_value, values))
