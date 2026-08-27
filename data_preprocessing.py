import pandas as pd
import numpy as np

# ============================================================
# DATA PREPROCESSOR
# ============================================================

class DataPreprocessor:
    """
    Handles data cleaning and preprocessing.

    The original DataFrame is not modified.
    All preprocessing is performed on a safe copy.
    """

    def __init__(self, df):
        self.df = df.copy()

        # Important columns preserved even if missing %
        # is greater than the threshold
        self.important_cols = [
            "EXT_SOURCE_1",
            "APARTMENTS_AVG",
            "BASEMENTAREA_AVG",
            "LIVINGAREA_AVG",
            "YEARS_BUILD_AVG",
            "FLOORSMAX_AVG",
            "ELEVATORS_AVG"
        ]

    # --------------------------------------------------------
    # 1. DROP HIGH MISSING VALUE COLUMNS
    # --------------------------------------------------------
    def drop_high_missing_columns(self, threshold=46):

        missing_percentage = (
            self.df.isnull()
            .mean()
            .mul(100)
        )

        cols_to_drop = missing_percentage[
            missing_percentage > threshold
        ].index.tolist()

        # Preserve important columns
        cols_to_drop = [
            col for col in cols_to_drop
            if col not in self.important_cols
        ]

        self.df.drop(
            columns=cols_to_drop,
            inplace=True
        )

        print(f"Missing Value Threshold: {threshold}%")
        print(f"Dropped Columns: {len(cols_to_drop)}")

        return self

    # --------------------------------------------------------
    # 2. CREATE EXT_SOURCE_1 MISSING INDICATOR
    # --------------------------------------------------------
    def create_ext_source_missing_indicator(self):

        if "EXT_SOURCE_1" in self.df.columns:

            self.df["EXT_SOURCE_1_MISSING"] = (
                self.df["EXT_SOURCE_1"]
                .isnull()
                .astype(int)
            )

        return self

    # --------------------------------------------------------
    # 3. HANDLE OCCUPATION_TYPE MISSING VALUES
    # --------------------------------------------------------
    def handle_occupation_type(self):

        if "OCCUPATION_TYPE" in self.df.columns:

            self.df["OCCUPATION_TYPE"] = (
                self.df["OCCUPATION_TYPE"]
                .fillna("Unknown")
            )

        return self

    # --------------------------------------------------------
    # 4. HANDLE NAME_TYPE_SUITE MISSING VALUES
    # --------------------------------------------------------
    def handle_name_type_suite(self):

        if "NAME_TYPE_SUITE" in self.df.columns:

            mode_value = (
                self.df["NAME_TYPE_SUITE"]
                .mode()[0]
            )

            self.df["NAME_TYPE_SUITE"] = (
                self.df["NAME_TYPE_SUITE"]
                .fillna(mode_value)
            )

        return self

    # --------------------------------------------------------
    # 5. HANDLE REMAINING NUMERICAL MISSING VALUES
    # --------------------------------------------------------
    def handle_numerical_missing_values(self):

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for col in numerical_columns:

            if self.df[col].isnull().sum() > 0:

                median_value = self.df[col].median()

                self.df[col] = (
                    self.df[col]
                    .fillna(median_value)
                )

        return self

    # --------------------------------------------------------
    # 6. HANDLE REMAINING CATEGORICAL MISSING VALUES
    # --------------------------------------------------------
    def handle_categorical_missing_values(self):

        categorical_columns = self.df.select_dtypes(
            include="object"
        ).columns

        for col in categorical_columns:

            if self.df[col].isnull().sum() > 0:

                mode_value = self.df[col].mode()[0]

                self.df[col] = (
                    self.df[col]
                    .fillna(mode_value)
                )

        return self

    # --------------------------------------------------------
    # 7. CHECK DAYS_EMPLOYED SPECIAL VALUE
    # --------------------------------------------------------
    def check_days_employed_special_value(self):

        if "DAYS_EMPLOYED" in self.df.columns:

            count = (
                self.df["DAYS_EMPLOYED"] == 365243
            ).sum()

            print(
                f"DAYS_EMPLOYED special value count: {count}"
            )

        return self

    # --------------------------------------------------------
    # 8. CAP CNT_CHILDREN
    # --------------------------------------------------------
    def cap_children(self, upper_limit=6):

        if "CNT_CHILDREN" in self.df.columns:

            self.df["CNT_CHILDREN"] = (
                self.df["CNT_CHILDREN"]
                .clip(upper=upper_limit)
            )

        return self

    # --------------------------------------------------------
    # 9. CAP OUTLIERS USING IQR METHOD
    # --------------------------------------------------------
    def cap_outliers(self):

        columns = [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE"
        ]

        for col in columns:

            if col in self.df.columns:

                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)

                IQR = Q3 - Q1

                lower_limit = Q1 - (1.5 * IQR)
                upper_limit = Q3 + (1.5 * IQR)

                self.df[col] = (
                    self.df[col]
                    .clip(
                        lower=lower_limit,
                        upper=upper_limit
                    )
                )

        return self

    # --------------------------------------------------------
    # 10. GET PREPROCESSED DATA
    # --------------------------------------------------------
    def get_data(self):

        return self.df.copy()