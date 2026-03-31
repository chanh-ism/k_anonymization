"""
Machine Learning Classification Performance.
"""

from enum import Enum

from numpy import average
from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,  # auc,; roc_curve,
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


class MLClassifierExample(Enum):
    """
    Set of example machine learning classifiers.

    This class serves as a convenient registry for common Scikit-Learn
    models. It is typically used in conjunction with the
    ``MLClassificationPerformance``.

    See Also
    --------
    MLClassificationPerformance
        Evaluation based on machine learning classification performance.
    """

    RF = RandomForestClassifier()
    """
    Random Forest.
    """

    LOG = LogisticRegression(max_iter=1000)
    """
    Logistic Regression.
    """

    SVM = SVC()
    """
    Support Vector Machine.
    """

    DT = DecisionTreeClassifier()
    """
    Decision Tree.
    """

    KNN = KNeighborsClassifier()
    """
    K-Nearest Neighbors.
    """

    def __get__(self, instance, owner):
        return self.value


class MLClassificationPerformance:
    """
    Evaluation based on machine learning classification performance.

    Parameters
    ----------
    model
        A machine learning classifier (e.g., RandomForest, SVC).
    df : DataFrame
        The data to be evaluated.
    feature_names : list
        A list of column names to be used as features (X).
    target_name : str
        The column name of the prediction target (y).
    split_ratio : float
        The fraction of the data to be used for testing (default 0.2).
    test_df : DataFrame
        An optional separate data to be used for testing. If provided,
        ``split_ratio`` is ignored and the whole ``df`` is used for training.
    seed : int
        Random seed for reproducibility in data splitting and model training.

    Attributes
    ----------
    X_train : DataFrame
        The processed training features.
    X_test : DataFrame
        The processed testing features.
    y_train : array-like
        The encoded training target labels.
    y_test : array-like
        The encoded testing target labels.
    metrics : dict
        A dictionary containing averaged accuracy, precision, recall, and F1.
    raw_metrics : dict
        A dictionary containing raw accuracy, precision, recall, and F1.
    classification_report : str
        The text report on the classification results
    confusion_matrix : ndarray
        The matrix showing true vs. predicted classifications.

    See Also
    --------
    MLClassifierExample
        Set of example machine learning classifiers.
    """

    def __init__(
        self,
        model,
        df: DataFrame,
        feature_names: list,
        target_name: str,
        split_ratio: float = 0.2,
        test_df: DataFrame = None,
        seed: int = None,
    ):
        """
        Initialize the machine learning classification evaluator.

        Parameters
        ----------
        model
            A machine learning classifier (e.g., RandomForest, SVC).
        df : DataFrame
            The data to be evaluated.
        feature_names : list
            A list of column names to be used as features (X).
        target_name : str
            The column name of the prediction target (y).
        split_ratio : float
            The fraction of the data to be used for testing (default 0.2).
        test_df : DataFrame
            An optional separate data to be used for testing. If provided,
            ``split_ratio`` is ignored and the whole ``df`` is used for training.
        seed : int
            Random seed for reproducibility in data splitting and model training.
        """
        self.seed = seed
        self.model = model
        self.split_ratio = split_ratio
        self.target = target_name
        self.features = feature_names

        self.validation_df = test_df
        self.test_df = test_df
        self.update_df(df)

    def update_df(self, df: DataFrame):
        """
        Change and preprocess the input data.

        This method performs the following:

        1. Categorical QIDs are One-Hot Encoded.

        2. Numerical QIDs are preserved.

        3. The target variable is Label Encoded.

        4. Data is split into training and testing sets if a ``test_df``
           is not presented.

        Parameters
        ----------
        df : DataFrame
            The data to be evaluated.
        """
        self._one_hot_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self._label_encoder = LabelEncoder()

        X_org = df[self.features]
        X_num = X_org.select_dtypes(include="number")
        X_cat = X_org.select_dtypes(exclude="number")

        self.X = concat(
            [
                X_num,
                DataFrame(
                    self._one_hot_encoder.fit_transform(X_cat),
                    columns=self._one_hot_encoder.get_feature_names_out(),
                ),
            ],
            axis=1,
        )
        self.y = self._label_encoder.fit_transform(df[self.target])
        if self.test_df is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=self.split_ratio,
                random_state=self.seed,
                stratify=self.y,
            )
        else:
            self.X_train = self.X
            self.y_train = self.y
            self._set_X_y_test_from_test_df()

        self.y_pred = None

    def _set_X_y_test_from_test_df(self):
        """
        Preprorocess the provided external test data.
        """
        if self.test_df is None:
            return
        X_org = self.test_df[self.features]
        X_num = X_org.select_dtypes(include="number")
        X_cat = X_org.select_dtypes(exclude="number")

        self.X_test = concat(
            [
                X_num,
                DataFrame(
                    self._one_hot_encoder.transform(X_cat),
                    columns=self._one_hot_encoder.get_feature_names_out(),
                ),
            ],
            axis=1,
        )
        self.y_test = self._label_encoder.transform(self.test_df[self.target])

    def _predict(self):
        """
        Train the model and generate predictions on the test set.
        """
        try:
            self.model.random_state = self.seed
        except:
            pass
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def _compute_metrics(self, y_test, y_pred, preview):
        """
        Calculate statistical performance metrics.

        Parameters
        ----------
        y_test
            Ground truth labels.
        y_pred
            Predicted labels.
        preview
            If True, prints a classification report.

        Returns
        -------
        tuple
            A tuple containing raw metrics (per class), averaged
            metrics, the text report, and the confusion matrix.
        """
        _y_test = self._label_encoder.inverse_transform(y_test)
        _y_pred = self._label_encoder.inverse_transform(y_pred)

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

    def evaluate(self, preview=False, restart=False):
        """
        Execute the model evaluation workflow.

        Fits the model (if not already trained), predicts, and
        stores classification results.

        Parameters
        ----------
        preview : bool, default False
            Whether to print the classification report.
        restart : bool, default False
            If True, forces the model to re-train and re-predict.
        """
        if restart or self.y_pred is None:
            self._predict()

        raw_metrics, metrics, report, pred_matrix = self._compute_metrics(
            self.y_test, self.y_pred, preview
        )
        self.raw_metrics = raw_metrics
        self.metrics = metrics
        self.classification_report = report
        self.confusion_matrix = pred_matrix
