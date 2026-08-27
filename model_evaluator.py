# ============================================================
# MODEL EVALUATOR
# ============================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    average_precision_score
)

from scipy.stats import ks_2samp


class ModelEvaluator:

    def __init__(
        self,
        model,
        model_name,
        X_test,
        y_test
    ):

        self.model = model
        self.model_name = model_name

        self.X_test = X_test.copy()
        self.y_test = y_test.copy()

        self.y_pred = None
        self.y_prob = None

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    def evaluate(self):

        self.y_pred = self.model.predict(
            self.X_test
        )

        self.y_prob = self.model.predict_proba(
            self.X_test
        )[:, 1]

        accuracy = accuracy_score(
            self.y_test,
            self.y_pred
        )

        precision = precision_score(
            self.y_test,
            self.y_pred,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            self.y_pred,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            self.y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            self.y_test,
            self.y_prob
        )

        print("=" * 60)
        print(f"MODEL EVALUATION: {self.model_name}")
        print("=" * 60)

        print("\nAccuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))
        print("ROC-AUC:", round(roc_auc, 4))

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                self.y_test,
                self.y_pred
            )
        )

        print("\nClassification Report:")

        print(
            classification_report(
                self.y_test,
                self.y_pred,
                zero_division=0
            )
        )

        return self


    # ========================================================
    # KS STATISTIC AND GINI COEFFICIENT
    # ========================================================

    def calculate_ks_gini(self):

        if self.y_prob is None:

            self.y_prob = self.model.predict_proba(
                self.X_test
            )[:, 1]

        default_probabilities = self.y_prob[
            self.y_test == 1
        ]

        non_default_probabilities = self.y_prob[
            self.y_test == 0
        ]

        ks_statistic = ks_2samp(
            default_probabilities,
            non_default_probabilities
        ).statistic

        roc_auc = roc_auc_score(
            self.y_test,
            self.y_prob
        )

        gini = (2 * roc_auc) - 1

        print("=" * 60)
        print(f"KS AND GINI: {self.model_name}")
        print("=" * 60)

        print(
            "\nKS Statistic:",
            round(ks_statistic, 4)
        )

        print(
            "ROC-AUC:",
            round(roc_auc, 4)
        )

        print(
            "Gini Coefficient:",
            round(gini, 4)
        )

        return self


    # ========================================================
    # CONFUSION MATRIX PLOT
    # ========================================================

    def plot_confusion_matrix(self):

        if self.y_pred is None:

            self.y_pred = self.model.predict(
                self.X_test
            )

        cm = confusion_matrix(
            self.y_test,
            self.y_pred
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        display.plot()

        plt.title(
            f"Confusion Matrix: {self.model_name}"
        )

        plt.show()

        return self


    # ========================================================
    # ROC CURVE
    # ========================================================

    def plot_roc_curve(self):

        if self.y_prob is None:

            self.y_prob = self.model.predict_proba(
                self.X_test
            )[:, 1]

        RocCurveDisplay.from_predictions(
            self.y_test,
            self.y_prob
        )

        plt.title(
            f"ROC Curve: {self.model_name}"
        )

        plt.show()

        return self


    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    def plot_precision_recall_curve(self):

        if self.y_prob is None:

            self.y_prob = self.model.predict_proba(
                self.X_test
            )[:, 1]

        precision, recall, _ = precision_recall_curve(
            self.y_test,
            self.y_prob
        )

        average_precision = average_precision_score(
            self.y_test,
            self.y_prob
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"Average Precision = "
                f"{average_precision:.4f}"
            )
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.title(
            f"Precision-Recall Curve: {self.model_name}"
        )

        plt.legend()
        plt.grid()

        plt.show()

        return self