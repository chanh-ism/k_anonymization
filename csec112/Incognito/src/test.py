from ucimlrepo import fetch_ucirepo
import pandas as pd
import utils, df_operations
from lattice import Lattice
from incognito import Incognito

# fetch dataset
adult = utils.fetch_dataset()


# # data (as pandas dataframes)
# X = adult.data.features
# y = adult.data.targets

# # metadata
# print(adult.metadata)

# # variable information
# print(adult.variables)

# print(adult.data)

# from utils import fetch_dataset_csv
# fetch_dataset_csv()


print("Reading generalization hierarchies for target quasi-identifiers...")
hierarchy = utils.read_hierarchies_by_col_names(
    [
        "workclass",
        "sex",
        "education",
        "marital-status",
        "age",
        "native-country",
        # 'occupation',
        # 'salary-class',
        # 'race',
    ]
)

hierarchy = hierarchy[
    (hierarchy["child_level"] == 0)
    &
    # (hierarchy['parent_level'] <= 2)&
    # (hierarchy['parent'] == 'Unemployed')&
    (True)
]
print("Generalization hierarchies read successfully.")
# print(hierarchy)
# generalized_df = df_operations.generalize(adult.data.original, hierarchy)
# k_ano = df_operations.is_k_anonymous(generalized_df, ['workclass'], 20)
# print(k_ano)

# lattice = Lattice(hierarchy)

# hierarchy内の準識別子について、生のadultをgroupbyしてカウント
# vc = adult.data.original[hierarchy['column'].unique()].value_counts(dropna=False)
# vc = vc.sort_index(level='workclass')  # workclass順
# print(vc)

print("Starting Incognito...")
# incognito_result = Incognito(adult, hierarchy, 20)
# incognito_result.verify_result()
utils.set_verbose(True)
adult = adult.dropna()
icg = Incognito_(adult, hierarchy, 2)
icg.run()
icg._print_result()
icg.verify_result()
# icg._print_result()
# icg._verify_result()
