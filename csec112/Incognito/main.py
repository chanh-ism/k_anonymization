import argparse

from src import utils
from src.incognito import Incognito
from src.utils import vprint

# parse command line arguments
parser = argparse.ArgumentParser(description="Run Incognito with specified parameters.")
parser.add_argument(
    "--dataset",
    type=str,
    default="adult",
    help="Dataset to use (default: 'adult'). Currently only 'adult' is supported.",
)
parser.add_argument(
    "--k",
    type=int,
    default=10,
    help="k-anonymity parameter (default: 20)",
)
parser.add_argument(
    "--q_cols",
    type=str,
    nargs="+",
    default=[
        "workclass",
        "sex",
        "education",
        "marital-status",
        # "age",
        "native-country",
    ],
    help="List of quasi-identifier columns to generalize (e.g., 'workclass', 'education').",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose output",
)
parser.add_argument(
    "--dropna",
    action="store_true",
    help="Drops records which includes NaN.",
)
parser.add_argument(
    "--size_limit",
    type=int,
    default=None,
    help="FOR DEBUG: Limit the size of the dataset to this number of records. If None, all records are used.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for results. If not specified, auto-generates from dataset, quasi-identifiers, k, and timestamp.",
)

args = parser.parse_args()
utils.set_verbose(args.verbose)

# read dataset
vprint("Reading dataset:", args.dataset)
dataset = utils.read_dataset(args.dataset)
vprint(f"Dataset loaded: {dataset.shape[0]} records.")

# limit dataset size if specified
if args.size_limit is not None:
    dataset = dataset.head(args.size_limit)
    vprint(f"Dataset limited into {dataset.shape[0]} records.")

# drop nan if specified
if args.dropna:
    vprint("Dropping records with NaN values...")
    dataset = utils.dropna(dataset)
    vprint(f"Records after dropping NaN: {dataset.shape[0]}")

# read hierarchies definition
vprint(f"Reading generalization hierarchies for {args.q_cols}...")
hierarchies_dir = f"Data/{args.dataset}/hierarchies"
hierarchy = utils.read_hierarchies_by_col_names(args.q_cols, hierarchies_dir)
# filter hierarchy
hierarchy = hierarchy[(hierarchy["child_level"] == 0)]

# incognito
print(f"Starting Incognito... with k={args.k} and quasi-identifiers: {args.q_cols}")
incognito = Incognito(dataset, hierarchy, args.k)
incognito.run()
incognito.print_result()
if utils.VERBOSE:
    incognito.verify_result()

# 出力ディレクトリの生成
if args.output:
    output_dir = args.output
else:
    from datetime import datetime
    q_cols_str = "_".join(args.q_cols)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"result/{args.dataset}_{q_cols_str}_k{args.k}_{timestamp}"

# 結果保存
incognito.save_result(output_dir)
