def get_max_ranges(data, qids_idx, is_cat, hierarchies):
    max_ranges = []

    columns = list(zip(*data))
    for pos, idx in enumerate(qids_idx):
        max_ranges.extend([None] * (idx - len(max_ranges)))
        if is_cat[pos] == True:
            max_ranges.append(
                len(hierarchies[idx]["lambda"])
                if hierarchies[idx]["type"] == "lambda"
                else len(hierarchies[idx]["hierarchy"])
            )
        else:
            max_ranges.append(max(columns[idx]) - min(columns[idx]))

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
        generalized_values = generalize(generalized_values, hierarchy, level)
        level += 1

    return level / height


def generalize(values, hierarchy, level):
    if hierarchy["type"] == "lambda":
        f = eval(hierarchy["lambda"][level])
    elif hierarchy["type"] == "list":
        is_suppressed = hierarchy["hierarchy"][level]["is_suppressed"]
        if is_suppressed:
            f = lambda x: "*"
        else:
            generalized_values = hierarchy["hierarchy"][level]["values"]

            def find_generalized_value(x):
                for generalized_value in generalized_values:
                    if x in generalized_value["original"]:
                        return generalized_value["generalized"]

            f = lambda x: find_generalized_value(x)
    return list(map(f, values))


def summarize(values, is_cat):
    anon_value = None
    if is_cat == True:
        try:
            anon_value = " ~ ".join(set(values))
        except:
            anon_value = " ~ ".join([str(x) for x in set(values)])
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
