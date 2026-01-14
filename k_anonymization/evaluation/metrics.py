# +
from enum import Enum

from numpy import average, ndarray
from pandas import concat, get_dummies
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  # auc,; roc_curve,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

from ..algorithms.type import Algorithm
from ..datasets import Dataset
from .utils import get_equivalence_qids


# -

class UtilityMetrics:
    @staticmethod
    def discernibility(
        equivalence_qids_counts: list,
        suppression_counts: int = 0,
        org_data_size: int = 0,
    ):
        return (
            sum([x**2 for x in equivalence_qids_counts])
            + suppression_counts * org_data_size
        )

    @staticmethod
    def discernibility_from_data(
        anon_data: DataFrame | ndarray,
        qids_idx: list,
        suppression_counts: int = 0,
        org_data_size: int = 0,
    ):
        equivalence_qids = get_equivalence_qids(anon_data, qids_idx)
        return UtilityMetrics.discernibility(
            [x["count"] for x in equivalence_qids], suppression_counts, org_data_size
        )

    @staticmethod
    def discernibility_from_algo(algo: Algorithm):
        return UtilityMetrics.discernibility_from_data(
            algo.anon_data,
            algo.dataset.qids_idx,
            (
                sum([x["count"] for x in algo.suppressed_qids])
                if algo.suppressed_qids
                else 0
            ),
            algo.org_data.shape[0],
        )

    @staticmethod
    def c_avg(
        equivalence_qids_counts: list,
        org_data_size: int,
        k: int,
    ):
        return float(org_data_size / (len(equivalence_qids_counts) * k))

    @staticmethod
    def c_avg_from_data(
        anon_data: DataFrame | ndarray,
        qids_idx: list,
        org_data_size: int,
        k: int,
    ):
        equivalence_qids = get_equivalence_qids(anon_data, qids_idx)
        return UtilityMetrics.c_avg(
            [x["count"] for x in equivalence_qids], org_data_size, k
        )

    @staticmethod
    def c_avg_from_algo(algo: Algorithm):
        return UtilityMetrics.c_avg_from_data(
            algo.anon_data,
            algo.dataset.qids_idx,
            algo.org_data.shape[0],
            algo.k,
        )

    @staticmethod
    def certainty_penalty_clusters_mean_mode(
        clusters: list,
        qids_idx: list,
        is_cat: list,
        max_ranges: list,
        org_data_size: int,
    ):
        _all_penalties = 0.0

        for cluster in clusters:
            _penalty = 0.0
            _size = len(cluster)
            columns = list(zip(*cluster))
            for pos, idx in enumerate(qids_idx):
                values = columns[idx]
                if is_cat[pos]:
                    mode = max(values, key=values.count)
                    _penalty += 1 - (values.count(mode) / _size)
                else:
                    _penalty += (max(values) - min(values)) / max_ranges[idx]
            _all_penalties += _penalty * _size
        return _all_penalties / (len(qids_idx) * org_data_size)

    @staticmethod
    def certainty_penalty_generalization(
        equivalence_qids: list,
        org_data_size: int,
        qids_idx: list,
        is_cat: list,
        hierarchies,
        max_ranges: list,
    ):
        _all_penalties = 0.0

        # TODO: Fix this! This only works with Datafly on ADULT
        # because all qids are suppressed except marital-status and occupation,
        # whose generalization trees are only of height 2
        def get_penalty_cat(value, hierarchy):
            if "values" not in list(hierarchy["tree"][0]):
                return 0
            all_leaves = 0.0
            for x in hierarchy["tree"][0]["values"]:
                if value in x["original"]:
                    return 0
                all_leaves += len(x["original"])

            for level in hierarchy["tree"][0:-1]:
                for x in level["values"]:
                    if value == x["generalized"]:
                        return len(x["original"]) / all_leaves

        for qid in equivalence_qids:
            _penalty = 0.0
            _size = qid["count"]
            for pos, val in enumerate(qid["qid"]):
                if val == "*":
                    _penalty += 1
                elif is_cat[pos]:
                    _penalty += get_penalty_cat(val, hierarchies[qids_idx[pos]])
                else:
                    try:
                        low, high = val.split("~")
                        _penalty += (float(high) - float(low)) / max_ranges[
                            qids_idx[pos]
                        ]
                    except:
                        continue

            _all_penalties += _penalty * _size
        return _all_penalties / (len(qids_idx) * org_data_size)


class MLClassifier(Enum):
    RF = RandomForestClassifier()
    LOG = LogisticRegression(max_iter=1000)
    SVM = SVC()
    DT = DecisionTreeClassifier()
    KNN = KNeighborsClassifier()
    XGB = XGBClassifier(n_estimators=100)

    def __get__(self, instance, owner):
        return self.value


class MLClassificationMetrics:
    def __init__(
        self,
        model,
        dataset: Dataset,
        features=[],
        df=None,
        seed=None,
        split_ratio=0.2,
        validation_df=None,
    ):
        self.seed = seed
        self.model = model
        self.split_ratio = split_ratio

        _df = df if df is not None else dataset.df
        self.target = dataset.target
        self.features = (
            [] if features != [] else [x for x in list(_df) if x != self.target]
        )
        self.label_encoder = LabelEncoder()
        self.validation_df = validation_df
        self.update_df(_df)
        if validation_df is not None:
            self.y_validation = self.label_encoder.fit_transform(
                validation_df[self.target]
            )

    def update_df(self, df):
        self.org_data = df
        self.one_hot_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )

        X_org = df[self.features]
        X_num = X_org.select_dtypes(exclude="object")
        X_cat = X_org.select_dtypes(include="object")

        self.X = concat(
            [
                X_num,
                DataFrame(
                    self.one_hot_encoder.fit_transform(X_cat),
                    columns=self.one_hot_encoder.get_feature_names_out(),
                ),
            ],
            axis=1,
        )
        self.y = self.label_encoder.fit_transform(df[self.target])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.split_ratio,
            random_state=self.seed,
            stratify=self.y,
        )
        self.y_pred = None
        self.set_X_validation()

    def set_X_validation(self):
        if self.validation_df is None:
            return

        X_org = self.validation_df[self.features]
        X_num = X_org.select_dtypes(exclude="object")
        X_cat = X_org.select_dtypes(include="object")

        self.X_validation = concat(
            [
                X_num,
                DataFrame(
                    self.one_hot_encoder.transform(X_cat),
                    columns=self.one_hot_encoder.get_feature_names_out(),
                ),
            ],
            axis=1,
        )
        self.y_pred_validation = None

    def predict(self):
        try:
            self.model.random_state = self.seed
        except:
            pass
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        if self.validation_df is not None:
            self.y_pred_validation = self.model.predict(self.X_validation)

    def __compute_metrics(self, y_test, y_pred, preview):
        _y_test = self.label_encoder.inverse_transform(y_test)
        _y_pred = self.label_encoder.inverse_transform(y_pred)

        raw_metrics = {
            "accuracy": accuracy_score(_y_test, _y_pred),
            "precision": precision_score(
                _y_test, _y_pred, average=None, zero_division=0
            ),
            "recall": recall_score(_y_test, _y_pred, average=None, zero_division=0),
            "f1_score": f1_score(_y_test, _y_pred, average=None, zero_division=0),
            # "auc": auc(fpr, tpr),
        }
        metrics = {
            "accuracy": raw_metrics["accuracy"],
            "precision": average(raw_metrics["precision"]).item(),
            "recall": average(raw_metrics["recall"]).item(),
            "f1_score": average(raw_metrics["f1_score"]).item(),
        }
        report = classification_report(_y_test, _y_pred, zero_division=0)
        pred_matrix = confusion_matrix(_y_test, _y_pred)
        if preview:
            print(report)
        return raw_metrics, metrics, report, pred_matrix

    def evaluate(self, preview=True, restart=False):
        if restart or self.y_pred is None:
            self.predict()

        raw_metrics, metrics, report, pred_matrix = self.__compute_metrics(
            self.y_test, self.y_pred, preview
        )
        self.raw_metrics = raw_metrics
        self.metrics = metrics
        self.classification_report = report
        self.confusion_matrix = pred_matrix

    def validate(self, preview=True, restart=False):
        if restart or self.y_pred_validation is None:
            self.predict()

        raw_metrics, metrics, report, pred_matrix = self.__compute_metrics(
            self.y_validation, self.y_pred_validation, preview
        )
        self.validation_raw_metrics = raw_metrics
        self.validation_metrics = metrics
        self.validation_classification_report = report
        self.validation_confusion_matrix = pred_matrix

    def evaluate_and_validate(self, preview=True, restart=False):
        self.evaluate(preview, restart)
        self.validate(preview)
