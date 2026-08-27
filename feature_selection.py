import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency

from sklearn.feature_selection import (
    mutual_info_classif,
    RFE
)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier


class FeatureSelector:
    """
    Performs feature selection for the Home Loan Scorecard.

    Methods:
    1. Correlation analysis
    2. Information Value (IV) / WOE
    3. Chi-Square test
    4. Mutual Information
    5. Recursive Feature Elimination (RFE)
    6. Random Forest Feature Importance

    Final feature selection follows the notebook:
    RFE selected features.
    """

    def __init__(self, X, y):

        self.X = X.copy()
        self.y = y.copy()

        self.X_corr_clean = None
        self.X_iv = None
        self.iv_data = None

        self.high_corr_df = None
        self.iv_results_df = None
        self.chi_square_results_df = None
        self.mi_results_df = None
        self.rfe_results_df = None
        self.rf_results_df = None

        self.iv_selected_features = None
        self.chi_square_selected_features = None
        self.mi_selected_features = None
        self.rfe_selected_features = None
        self.rf_selected_features = None

        self.X_final = None
        self.y_final = None

    # ========================================================
    # STEP 7.1: CORRELATION ANALYSIS
    # ========================================================

    def correlation_analysis(self, threshold=0.90):

        X_numeric = self.X.select_dtypes(
            include=np.number
        )

        corr_matrix = X_numeric.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(bool)
        )

        high_corr_pairs = []

        for column in upper_triangle.columns:

            correlated_features = upper_triangle.index[
                upper_triangle[column] > threshold
            ].tolist()

            for feature in correlated_features:

                high_corr_pairs.append(
                    (
                        feature,
                        column,
                        upper_triangle.loc[
                            feature,
                            column
                        ]
                    )
                )

        self.high_corr_df = pd.DataFrame(
            high_corr_pairs,
            columns=[
                "Feature_1",
                "Feature_2",
                "Correlation"
            ]
        )

        if not self.high_corr_df.empty:
            self.high_corr_df = (
                self.high_corr_df
                .sort_values(
                    by="Correlation",
                    ascending=False
                )
                .reset_index(drop=True)
            )

        print(
            "Number of Highly Correlated Pairs:",
            len(self.high_corr_df)
        )

        return self

    # ========================================================
    # STEP 7.1: REMOVE REDUNDANT FEATURES
    # ========================================================

    def remove_redundant_features(self):

        features_to_drop = [
            "DAYS_REGISTRATION",
            "FLAG_OWN_CAR",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
            "FLAG_OWN_REALTY",
            "DAYS_BIRTH",
            "AGE",
            "DAYS_EMPLOYED"
        ]

        existing_features_to_drop = [
            col
            for col in features_to_drop
            if col in self.X.columns
        ]

        self.X_corr_clean = self.X.drop(
            columns=existing_features_to_drop
        ).copy()

        print(
            "Original X Shape:",
            self.X.shape
        )

        print(
            "Features Removed:",
            len(existing_features_to_drop)
        )

        print(
            "Cleaned X Shape:",
            self.X_corr_clean.shape
        )

        print("\nRemoved Features:")
        print(existing_features_to_drop)

        return self

    # ========================================================
    # STEP 7.2: PREPARE IV DATA
    # ========================================================

    def prepare_iv_data(self):

        if self.X_corr_clean is None:
            raise ValueError(
                "Run remove_redundant_features() first."
            )

        self.X_iv = self.X_corr_clean.copy()

        self.iv_data = self.X_iv.copy()

        self.iv_data["TARGET"] = self.y.values

        print(
            "Feature Matrix Shape:",
            self.X_iv.shape
        )

        print(
            "IV Dataset Shape:",
            self.iv_data.shape
        )

        return self

    # ========================================================
    # WOE / IV CALCULATION
    # ========================================================

    @staticmethod
    def calculate_woe_iv(
        data,
        feature,
        target="TARGET",
        bins=10
    ):

        df = data[
            [feature, target]
        ].copy()

        if df[feature].nunique() > 10:

            try:
                df["BIN"] = pd.qcut(
                    df[feature],
                    q=bins,
                    duplicates="drop"
                )

            except ValueError:
                df["BIN"] = df[feature]

        else:
            df["BIN"] = df[feature]

        df["BIN"] = df["BIN"].astype(object)

        df.loc[
            df[feature].isnull(),
            "BIN"
        ] = "Missing"

        grouped = (
            df.groupby(
                "BIN",
                observed=False
            )[target]
            .agg(
                Total="count",
                Bad="sum"
            )
        )

        grouped["Good"] = (
            grouped["Total"]
            - grouped["Bad"]
        )

        grouped["Good"] = (
            grouped["Good"] + 0.5
        )

        grouped["Bad"] = (
            grouped["Bad"] + 0.5
        )

        grouped["Dist_Good"] = (
            grouped["Good"]
            / grouped["Good"].sum()
        )

        grouped["Dist_Bad"] = (
            grouped["Bad"]
            / grouped["Bad"].sum()
        )

        grouped["WOE"] = np.log(
            grouped["Dist_Good"]
            / grouped["Dist_Bad"]
        )

        grouped["IV_Component"] = (
            grouped["Dist_Good"]
            - grouped["Dist_Bad"]
        ) * grouped["WOE"]

        iv_value = (
            grouped["IV_Component"].sum()
        )

        return iv_value, grouped

    # ========================================================
    # STEP 7.2: CALCULATE IV
    # ========================================================

    def calculate_iv(self):

        if self.iv_data is None:
            raise ValueError(
                "Run prepare_iv_data() first."
            )

        iv_results = []

        for feature in self.X_iv.columns:

            iv_value, _ = self.calculate_woe_iv(
                data=self.iv_data,
                feature=feature,
                target="TARGET",
                bins=10
            )

            iv_results.append({
                "Feature": feature,
                "IV": iv_value
            })

        self.iv_results_df = pd.DataFrame(
            iv_results
        )

        self.iv_results_df = (
            self.iv_results_df
            .sort_values(
                by="IV",
                ascending=False
            )
            .reset_index(drop=True)
        )

        self.iv_results_df[
            "IV_Interpretation"
        ] = pd.cut(
            self.iv_results_df["IV"],
            bins=[
                -np.inf,
                0.02,
                0.10,
                0.30,
                np.inf
            ],
            labels=[
                "Weak",
                "Medium",
                "Strong",
                "Very Strong"
            ]
        )

        print(
            "Total Features Evaluated:",
            len(self.iv_results_df)
        )

        return self

    # ========================================================
    # STEP 7.2: SELECT IV FEATURES
    # ========================================================

    def select_iv_features(self, threshold=0.02):

        if self.iv_results_df is None:
            raise ValueError(
                "Run calculate_iv() first."
            )

        self.iv_selected_features = (
            self.iv_results_df.loc[
                self.iv_results_df["IV"] >= threshold,
                "Feature"
            ].tolist()
        )

        print(
            f"Features with IV >= {threshold}:",
            len(self.iv_selected_features)
        )

        return self

    # ========================================================
    # STEP 7.3: CHI-SQUARE TEST
    # ========================================================

    def chi_square_test(
        self,
        max_unique_values=10,
        significance_level=0.05
    ):

        if self.X_corr_clean is None:
            raise ValueError(
                "Run remove_redundant_features() first."
            )

        chi_square_features = []

        for col in self.X_corr_clean.columns:

            n_unique = (
                self.X_corr_clean[col].nunique()
            )

            if n_unique <= max_unique_values:
                chi_square_features.append(col)

        chi_square_results = []

        for feature in chi_square_features:

            contingency_table = pd.crosstab(
                self.X_corr_clean[feature],
                self.y
            )

            if (
                contingency_table.shape[0] < 2
                or contingency_table.shape[1] < 2
            ):
                continue

            chi2, p_value, dof, expected = (
                chi2_contingency(
                    contingency_table
                )
            )

            chi_square_results.append({
                "Feature": feature,
                "Chi_Square": chi2,
                "P_Value": p_value,
                "Degrees_of_Freedom": dof
            })

        self.chi_square_results_df = pd.DataFrame(
            chi_square_results
        )

        if not self.chi_square_results_df.empty:

            self.chi_square_results_df[
                "Significant"
            ] = (
                self.chi_square_results_df["P_Value"]
                < significance_level
            )

            self.chi_square_results_df = (
                self.chi_square_results_df
                .sort_values(
                    by="Chi_Square",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            self.chi_square_selected_features = (
                self.chi_square_results_df.loc[
                    self.chi_square_results_df["P_Value"]
                    < significance_level,
                    "Feature"
                ].tolist()
            )

        else:

            self.chi_square_selected_features = []

        print(
            "Total Features Tested:",
            len(self.chi_square_results_df)
        )

        print(
            "Significant Features:",
            len(
                self.chi_square_selected_features
            )
        )

        return self

    # ========================================================
    # STEP 7.4: MUTUAL INFORMATION
    # ========================================================

    def mutual_information(self, top_n=40):

        if self.X_corr_clean is None:
            raise ValueError(
                "Run remove_redundant_features() first."
            )

        discrete_mask = (
            self.X_corr_clean
            .nunique()
            <= 10
        )

        mi_scores = mutual_info_classif(
            self.X_corr_clean,
            self.y,
            discrete_features=discrete_mask,
            random_state=42,
            n_neighbors=3
        )

        self.mi_results_df = pd.DataFrame({
            "Feature":
                self.X_corr_clean.columns,
            "Mutual_Information":
                mi_scores
        })

        self.mi_results_df = (
            self.mi_results_df
            .sort_values(
                by="Mutual_Information",
                ascending=False
            )
            .reset_index(drop=True)
        )

        self.mi_selected_features = (
            self.mi_results_df
            .head(top_n)["Feature"]
            .tolist()
        )

        print(
            "Selected Top MI Features:",
            len(
                self.mi_selected_features
            )
        )

        return self

    # ========================================================
    # STEP 7.5: RFE
    # ========================================================

    def recursive_feature_elimination(
        self,
        sample_size=50000,
        n_features_to_select=40,
        step=5
    ):

        if self.X_corr_clean is None:
            raise ValueError(
                "Run remove_redundant_features() first."
            )

        X_rfe_sample, _, y_rfe_sample, _ = (
            train_test_split(
                self.X_corr_clean,
                self.y,
                train_size=sample_size,
                stratify=self.y,
                random_state=42
            )
        )

        scaler_rfe = StandardScaler()

        X_rfe_scaled = scaler_rfe.fit_transform(
            X_rfe_sample
        )

        X_rfe_scaled = pd.DataFrame(
            X_rfe_scaled,
            columns=X_rfe_sample.columns,
            index=X_rfe_sample.index
        )

        log_reg_rfe = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )

        rfe_selector = RFE(
            estimator=log_reg_rfe,
            n_features_to_select=n_features_to_select,
            step=step
        )

        rfe_selector.fit(
            X_rfe_scaled,
            y_rfe_sample
        )

        self.rfe_results_df = pd.DataFrame({
            "Feature":
                X_rfe_scaled.columns,
            "Selected":
                rfe_selector.support_,
            "Ranking":
                rfe_selector.ranking_
        })

        self.rfe_results_df = (
            self.rfe_results_df
            .sort_values(
                by=[
                    "Selected",
                    "Ranking"
                ],
                ascending=[
                    False,
                    True
                ]
            )
            .reset_index(drop=True)
        )

        self.rfe_selected_features = (
            self.rfe_results_df.loc[
                self.rfe_results_df["Selected"],
                "Feature"
            ].tolist()
        )

        self.X_rfe_sample = X_rfe_sample
        self.y_rfe_sample = y_rfe_sample

        print(
            "Total Features:",
            len(self.rfe_results_df)
        )

        print(
            "Selected RFE Features:",
            len(
                self.rfe_selected_features
            )
        )

        return self

    # ========================================================
    # STEP 7.6: RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    def random_forest_importance(self, top_n=40):

        if not hasattr(
            self,
            "X_rfe_sample"
        ):
            raise ValueError(
                "Run recursive_feature_elimination() first."
            )

        rf_feature_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        rf_feature_model.fit(
            self.X_rfe_sample,
            self.y_rfe_sample
        )

        self.rf_feature_model = rf_feature_model

        self.rf_results_df = pd.DataFrame({
            "Feature":
                self.X_rfe_sample.columns,
            "Importance":
                rf_feature_model.feature_importances_
        })

        self.rf_results_df = (
            self.rf_results_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        self.rf_selected_features = (
            self.rf_results_df
            .head(top_n)["Feature"]
            .tolist()
        )

        print(
            "Selected Random Forest Features:",
            len(
                self.rf_selected_features
            )
        )

        return self

    # ========================================================
    # STEP 7.8: FINAL FEATURE SELECTION
    # ========================================================

    def create_final_features(self):

        if self.rfe_selected_features is None:
            raise ValueError(
                "Run recursive_feature_elimination() first."
            )

        final_features = (
            self.rfe_selected_features.copy()
        )

        self.X_final = (
            self.X_corr_clean[
                final_features
            ].copy()
        )

        self.y_final = self.y.copy()

        print(
            "Number of Final Selected Features:",
            len(final_features)
        )

        print(
            "Final Feature Dataset Shape:",
            self.X_final.shape
        )

        print(
            "Final Target Shape:",
            self.y_final.shape
        )

        return self

    # ========================================================
    # GET FINAL DATA
    # ========================================================

    def get_final_data(self):

        if self.X_final is None:
            raise ValueError(
                "Run create_final_features() first."
            )

        return (
            self.X_final.copy(),
            self.y_final.copy()
        )