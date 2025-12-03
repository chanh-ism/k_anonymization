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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

from ..algorithms.type import Algorithm
from ..datasets import Dataset
from .utils import count_equivalent_qids


# -

def discernibility(
    equivalent_qids: list,
    suppression_counts: int = 0,
    org_data_size: int = 0,
):
    return sum([x**2 for x in equivalent_qids]) + suppression_counts * org_data_size


def discernibility_from_data(
    anon_data: DataFrame | ndarray,
    qids_idx: list = [],
    suppression_counts: int = 0,
    org_data_size: int = 0,
):
    equivalent_qids = count_equivalent_qids(anon_data, qids_idx=qids_idx)
    return discernibility(equivalent_qids, suppression_counts, org_data_size)


def discernibility_from_algo(algo: Algorithm):
    return discernibility_from_data(
        algo.anon_data,
        algo.dataset.qids_idx,
        (
            sum([x["count"] for x in algo.suppressed_qids])
            if algo.suppressed_qids
            else 0
        ),
        algo.org_data.shape[0],
    )


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
    ):
        self.seed = seed
        self.model = model
        self.split_ratio = split_ratio

        _df = df if df != None else dataset.df
        self.target = dataset.target
        self.features = (
            [] if features != [] else [x for x in list(_df) if x != self.target]
        )
        self.label_encoder = LabelEncoder()
        self.update_df(_df)

    def update_df(self, df):
        self.org_data = df

        X_org = df[self.features]
        X_num = X_org.select_dtypes(exclude="object")
        X_cat = X_org.select_dtypes(include="object")

        self.X = concat([X_num, get_dummies(X_cat, dtype=int, drop_first=True)], axis=1)
        self.y = self.label_encoder.fit_transform(df[self.target])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.split_ratio,
            random_state=self.seed,
            stratify=self.y,
        )
        self.y_pred = None

    def predict(self):
        try:
            self.model.random_state = self.seed
        except:
            pass
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self, preview=True, restart=False):
        if restart or self.y_pred is None:
            self.predict()

        # fpr, tpr, _ = roc_curve(self.y_test, self.y_pred, p)

        y_test = self.label_encoder.inverse_transform(self.y_test)
        y_pred = self.label_encoder.inverse_transform(self.y_pred)

        self.raw_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=None, zero_division=0),
            "recall": recall_score(y_test, y_pred, average=None, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average=None, zero_division=0),
            # "auc": auc(fpr, tpr),
        }
        self.metrics = {
            "accuracy": self.raw_metrics["accuracy"],
            "precision": average(self.raw_metrics["precision"]).item(),
            "recall": average(self.raw_metrics["recall"]).item(),
            "f1_score": average(self.raw_metrics["f1_score"]).item(),
        }
        self.classification_report = classification_report(
            y_test, y_pred, zero_division=0
        )
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        if preview:
            print(self.classification_report)
