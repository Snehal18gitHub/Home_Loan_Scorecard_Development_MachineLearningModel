# ============================================================
# MODEL CROSS VALIDATOR
# ============================================================

import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate


class ModelCrossValidator:

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(
        self,
        model,
        model_name,
        X,
        y,
        n_splits=5
    ):

        self.model = model
        self.model_name = model_name

        # Safe copies
        self.X = X.copy()
        self.y = y.copy()

        self.n_splits = n_splits

        self.results = None

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    def run_cross_validation(self):

        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42
        )

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        }

        self.results = cross_validate(
            estimator=self.model,
            X=self.X,
            y=self.y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        print("=" * 60)
        print(f"CROSS VALIDATION: {self.model_name}")
        print("=" * 60)

        print(
            "\nAccuracy:",
            round(
                self.results["test_accuracy"].mean(),
                4
            )
        )

        print(
            "Precision:",
            round(
                self.results["test_precision"].mean(),
                4
            )
        )

        print(
            "Recall:",
            round(
                self.results["test_recall"].mean(),
                4
            )
        )

        print(
            "F1 Score:",
            round(
                self.results["test_f1"].mean(),
                4
            )
        )

        print(
            "ROC-AUC:",
            round(
                self.results["test_roc_auc"].mean(),
                4
            )
        )

        return self

    # ========================================================
    # CROSS VALIDATION SUMMARY
    # ========================================================

    def get_summary(self):

        if self.results is None:

            raise ValueError(
                "Please run cross validation first."
            )

        summary = pd.DataFrame({

            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],

            "Mean Score": [
                self.results["test_accuracy"].mean(),
                self.results["test_precision"].mean(),
                self.results["test_recall"].mean(),
                self.results["test_f1"].mean(),
                self.results["test_roc_auc"].mean()
            ],

            "Standard Deviation": [
                self.results["test_accuracy"].std(),
                self.results["test_precision"].std(),
                self.results["test_recall"].std(),
                self.results["test_f1"].std(),
                self.results["test_roc_auc"].std()
            ]

        })

        return summary.round(4)