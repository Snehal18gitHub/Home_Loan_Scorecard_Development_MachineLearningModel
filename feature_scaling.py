# ============================================================
# FEATURE SCALER
# ============================================================

import pandas as pd

from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """
    Handles standard feature scaling.

    The original feature matrix is not modified.
    Only the selected continuous numerical features
    are scaled.
    """

    def __init__(self, X):

        # Safe copy
        self.X = X.copy()

        # StandardScaler
        self.scaler = StandardScaler()

        # Exact columns from the original notebook
        self.scale_cols = [

            # Applicant / Family
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "CNT_FAM_MEMBERS",

            # Loan Information
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",

            # Regional / Time Variables
            "REGION_POPULATION_RELATIVE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
            "HOUR_APPR_PROCESS_START",
            "DAYS_LAST_PHONE_CHANGE",

            # Credit Bureau
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",

            # Engineered Continuous Features
            "AGE_YEARS",
            "EMPLOYMENT_YEARS",
            "LTI_RATIO",
            "EMI_INCOME_RATIO",
            "LTV_RATIO",
            "DISPOSABLE_MONTHLY_INCOME",
            "REGISTRATION_YEARS",
            "TOTAL_BUREAU_INQUIRIES",
            "RECENT_BUREAU_INQUIRIES",

            # External Scores
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "AVG_EXTERNAL_SCORE",
            "MAX_EXTERNAL_SCORE",
            "MIN_EXTERNAL_SCORE",

            # Property Features
            "APARTMENTS_AVG",
            "BASEMENTAREA_AVG",
            "YEARS_BUILD_AVG",
            "ELEVATORS_AVG",
            "FLOORSMAX_AVG",
            "LIVINGAREA_AVG",
            "PROPERTY_QUALITY_SCORE",

            # Social / Risk Count Features
            "OBS_30_CNT_SOCIAL_CIRCLE",
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "OBS_60_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE",
            "SOCIAL_RISK_SCORE"
        ]

    # --------------------------------------------------------
    # STEP 6.1: IDENTIFY NUMERICAL FEATURES
    # --------------------------------------------------------
    def get_numerical_columns(self):

        numerical_cols = self.X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        print(
            "Total Numerical Features:",
            len(numerical_cols)
        )

        print("\nNumerical Features:")
        print(numerical_cols)

        return numerical_cols

    # --------------------------------------------------------
    # STEP 6.2: SHOW FEATURES TO BE SCALED
    # --------------------------------------------------------
    def show_scale_columns(self):

        print(
            "Number of Features to Scale:",
            len(self.scale_cols)
        )

        print(self.scale_cols)

        return self

    # --------------------------------------------------------
    # STEP 6.3: STANDARD FEATURE SCALING
    # --------------------------------------------------------
    def scale_features(self):

        X_scaled = self.X.copy()

        X_scaled[self.scale_cols] = (
            self.scaler.fit_transform(
                X_scaled[self.scale_cols]
            )
        )

        self.X_scaled = X_scaled

        print(
            "Feature Scaling Completed Successfully!"
        )

        print(
            "Original X Shape:",
            self.X.shape
        )

        print(
            "Scaled X Shape:",
            self.X_scaled.shape
        )

        return self

    # --------------------------------------------------------
    # STEP 6.4: SCALING VALIDATION
    # --------------------------------------------------------
    def validate_scaling(self):

        scaling_check = pd.DataFrame({
            "Mean": (
                self.X_scaled[self.scale_cols]
                .mean()
            ),
            "Std": (
                self.X_scaled[self.scale_cols]
                .std()
            )
        }).round(2)

        return scaling_check

    # --------------------------------------------------------
    # GET SCALED DATA
    # --------------------------------------------------------
    def get_data(self):

        return self.X_scaled.copy()