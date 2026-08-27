import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SHAP ANALYZER
# ============================================================

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SHAPAnalyzer:

    def __init__(
        self,
        model,
        X_train,
        X_test,
        sample_size=5000,
        random_state=42
    ):

        self.model = model
        self.X_train = X_train
        self.X_test = X_test

        self.sample_size = sample_size
        self.random_state = random_state

        self.X_shap_sample = None
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

    # ========================================================
    # 1. CREATE REPRESENTATIVE SAMPLE
    # ========================================================

    def create_sample(self):

        sample_size = min(
            self.sample_size,
            len(self.X_train)
        )

        self.X_shap_sample = self.X_train.sample(
            n=sample_size,
            random_state=self.random_state
        )

        print("SHAP Sample Created Successfully!")
        print("SHAP Sample Shape:", self.X_shap_sample.shape)

        return self.X_shap_sample

    # ========================================================
    # 2. CREATE SHAP EXPLAINER
    # ========================================================

    def create_explainer(self):

        if self.X_shap_sample is None:
            self.create_sample()

        self.explainer = shap.TreeExplainer(
            self.model
        )

        print("SHAP TreeExplainer Created Successfully!")

        return self.explainer

    # ========================================================
    # 3. CALCULATE SHAP VALUES
    # ========================================================

    def calculate_shap_values(self):

        if self.explainer is None:
            self.create_explainer()

        raw_shap_values = self.explainer(
            self.X_shap_sample
        )

        # ----------------------------------------------------
        # Handle binary classification output
        # ----------------------------------------------------

        if raw_shap_values.values.ndim == 3:

            # Class 1 = Default
            self.shap_values = shap.Explanation(
                values=raw_shap_values.values[:, :, 1],
                base_values=raw_shap_values.base_values[:, 1],
                data=raw_shap_values.data,
                feature_names=raw_shap_values.feature_names
            )

        else:

            self.shap_values = raw_shap_values

        print("SHAP Values Calculated Successfully!")
        print("SHAP Values Shape:", self.shap_values.values.shape)

        return self.shap_values

    # ========================================================
    # 4. SHAP SUMMARY PLOT
    # ========================================================

    def summary_plot(self):

        if self.shap_values is None:
            self.calculate_shap_values()

        print("Generating SHAP Summary Plot...")

        shap.plots.beeswarm(
            self.shap_values,
            max_display=20,
            show=True
        )

    # ========================================================
    # 5. SHAP FEATURE IMPORTANCE
    # ========================================================

    def feature_importance(self):

        if self.shap_values is None:
            self.calculate_shap_values()

        importance = np.abs(
            self.shap_values.values
        ).mean(axis=0)

        feature_importance_df = pd.DataFrame({

            "Feature":
                self.X_shap_sample.columns,

            "Mean_Absolute_SHAP":
                importance

        })

        feature_importance_df = (
            feature_importance_df
            .sort_values(
                by="Mean_Absolute_SHAP",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print("\nTop 20 SHAP Features:")

        print(feature_importance_df.head(20))

        return feature_importance_df

    # ========================================================
    # 6. SHAP FEATURE IMPORTANCE BAR PLOT
    # ========================================================

    def feature_importance_plot(self):

        if self.shap_values is None:
            self.calculate_shap_values()

        shap.plots.bar(
            self.shap_values,
            max_display=20,
            show=True
        )

    # ========================================================
    # 7. SHAP DEPENDENCE PLOT
    # ========================================================

    def dependence_plot(self, feature):

        if self.shap_values is None:
            self.calculate_shap_values()

        shap.plots.scatter(
            self.shap_values[:, feature],
            color=self.shap_values,
            show=True
        )

    # ========================================================
    # 8. EXPLAIN ONE APPLICANT
    # ========================================================

    def explain_single_prediction(self, X_instance):

        if self.explainer is None:
            self.create_explainer()

        raw_values = self.explainer(X_instance)

        # Handle binary classification
        if raw_values.values.ndim == 3:

            instance_shap = shap.Explanation(
                values=raw_values.values[:, :, 1],
                base_values=raw_values.base_values[:, 1],
                data=raw_values.data,
                feature_names=raw_values.feature_names
            )

        else:

            instance_shap = raw_values

        shap.plots.waterfall(
            instance_shap[0],
            max_display=15
        )

        return instance_shap