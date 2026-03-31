from k_anonymization.core import Dataset


def get_max_ranges(dataset: Dataset):
    qids_idx = dataset.qids_idx
    is_categorical = dataset.is_categorical
    hierarchies = dataset.hierarchies
    df = dataset.df
    columns = list(df)

    max_ranges = []
    for pos, idx in enumerate(qids_idx):
        max_ranges.extend([None] * (idx - len(max_ranges)))
        if is_categorical[pos]:
            max_ranges.append(hierarchies[idx].height)
        else:
            max_ranges.append(df[columns[idx]].max() - df[columns[idx]].min())

    max_ranges.extend([None] * (len(columns) - len(max_ranges)))

    return max_ranges


def get_distance(r, record, qids_idx, is_categorical, max_ranges, hierarchies):
    distances = []

    for pos, idx in enumerate(qids_idx):
        if is_categorical[pos]:
            distances.append(
                get_categorical_distance([r[idx], record[idx]], hierarchies[idx])
            )
        else:
            distances.append(abs(r[idx] - record[idx]) / max_ranges[idx])

    return sum(distances)


def get_information_loss(
    record, cluster, qids_idx, is_categorical, max_ranges, hierarchies
):
    information_losses = []
    if record is None:
        size = len(cluster)
        columns = list(zip(*cluster))
    else:
        size = len(cluster) + 1
        columns = list(zip(*(cluster + [record])))

    for pos, idx in enumerate(qids_idx):
        if is_categorical[pos]:
            information_losses.append(
                get_categorical_distance(columns[idx], hierarchies[idx])
            )
        else:
            information_losses.append(
                (max(columns[idx]) - min(columns[idx])) / max_ranges[idx]
            )

    return size * sum(information_losses)


def get_categorical_distance(values, hierarchy):
    _level = hierarchy.get_lowest_common_ancestor(values, get_type="height")
    return _level / hierarchy.height
