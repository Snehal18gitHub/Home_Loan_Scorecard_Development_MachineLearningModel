import pandas as pd
import numpy as np


class InferencePreprocessor:
    """
    Prepares a single new applicant for prediction.

    The output is guaranteed to contain the exact
    40 features expected by the trained model.
    """

    def __init__(self, feature_names):

        self.feature_names = feature_names

    # ========================================================
    # CREATE APPLICANT FEATURES
    # ========================================================

    def transform(self, applicant):

        df = pd.DataFrame([applicant])

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        if "DAYS_BIRTH" in df.columns:

            df["AGE_YEARS"] = (
                abs(df["DAYS_BIRTH"]) / 365
            ).astype(int)

        # ----------------------------------------------------
        # EMPLOYMENT
        # ----------------------------------------------------

        if "DAYS_EMPLOYED" in df.columns:

            df["EMPLOYMENT_DATA_UNAVAILABLE"] = np.where(
                df["DAYS_EMPLOYED"] == 365243,
                1,
                0
            )

            df["EMPLOYMENT_YEARS"] = np.where(
                df["DAYS_EMPLOYED"] == 365243,
                0,
                abs(df["DAYS_EMPLOYED"]) / 365
            )

        # ----------------------------------------------------
        # INCOME CATEGORY
        # ----------------------------------------------------

        if "AMT_INCOME_TOTAL" in df.columns:

            income = df["AMT_INCOME_TOTAL"]

            df["INCOME_CATEGORY"] = np.select(
                [
                    income <= 112500,
                    income.between(112500, 147150),
                    income.between(147150, 202500),
                    income > 202500
                ],
                [
                    0,
                    1,
                    2,
                    3
                ],
                default=0
            )

        # ----------------------------------------------------
        # LOAN FEATURES
        # ----------------------------------------------------

        if {
            "AMT_CREDIT",
            "AMT_INCOME_TOTAL"
        }.issubset(df.columns):

            df["LTI_RATIO"] = (
                df["AMT_CREDIT"]
                / df["AMT_INCOME_TOTAL"]
            )

        if {
            "AMT_ANNUITY",
            "AMT_INCOME_TOTAL"
        }.issubset(df.columns):

            df["EMI_INCOME_RATIO"] = (
                df["AMT_ANNUITY"]
                / (df["AMT_INCOME_TOTAL"] / 12)
            )

        if {
            "AMT_CREDIT",
            "AMT_GOODS_PRICE"
        }.issubset(df.columns):

            df["LTV_RATIO"] = (
                df["AMT_CREDIT"]
                / df["AMT_GOODS_PRICE"]
            )

        # ----------------------------------------------------
        # VEHICLE OWNERSHIP
        # ----------------------------------------------------

        if "FLAG_OWN_CAR" in df.columns:

            df["OWNS_VEHICLE"] = (
                df["FLAG_OWN_CAR"]
                .map({
                    "N": 0,
                    "Y": 1
                })
                .fillna(0)
            )

        # ----------------------------------------------------
        # EXTERNAL CREDIT SCORES
        # ----------------------------------------------------

        external_columns = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        ]

        existing_external = [
            col
            for col in external_columns
            if col in df.columns
        ]

        if existing_external:

            df["AVG_EXTERNAL_SCORE"] = (
                df[existing_external]
                .mean(axis=1)
            )

            df["MAX_EXTERNAL_SCORE"] = (
                df[existing_external]
                .max(axis=1)
            )

            df["MIN_EXTERNAL_SCORE"] = (
                df[existing_external]
                .min(axis=1)
            )

        # ----------------------------------------------------
        # EXT SOURCE 1 MISSING FLAG
        # ----------------------------------------------------

        if "EXT_SOURCE_1" in df.columns:

            df["EXT_SOURCE_1_MISSING"] = (
                df["EXT_SOURCE_1"]
                .isnull()
                .astype(int)
            )

        # ----------------------------------------------------
        # PROPERTY QUALITY
        # ----------------------------------------------------

        property_columns = [
            "APARTMENTS_AVG",
            "BASEMENTAREA_AVG",
            "LIVINGAREA_AVG",
            "YEARS_BUILD_AVG",
            "FLOORSMAX_AVG",
            "ELEVATORS_AVG"
        ]

        existing_property = [
            col
            for col in property_columns
            if col in df.columns
        ]

        if existing_property:

            df["PROPERTY_QUALITY_SCORE"] = (
                df[existing_property]
                .mean(axis=1)
            )

        # ----------------------------------------------------
        # DOCUMENT COMPLETENESS
        # ----------------------------------------------------

        document_columns = [
            col
            for col in df.columns
            if col.startswith("FLAG_DOCUMENT")
        ]

        if document_columns:

            df["DOCUMENT_COMPLETENESS_SCORE"] = (
                df[document_columns]
                .sum(axis=1)
            )

        # ----------------------------------------------------
        # GENDER ENCODING
        # ----------------------------------------------------

        if "CODE_GENDER" in df.columns:

            df["CODE_GENDER"] = (
                df["CODE_GENDER"]
                .map({
                    "F": 0,
                    "M": 1,
                    "XNA": -1
                })
            )

        # ----------------------------------------------------
        # INCOME TYPE ONE-HOT
        # ----------------------------------------------------

        income_types = [
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Working"
        ]

        if "NAME_INCOME_TYPE" in df.columns:

            for income_type in income_types:

                column_name = (
                    "NAME_INCOME_TYPE_"
                    + income_type
                )

                df[column_name] = (
                    df["NAME_INCOME_TYPE"]
                    == income_type
                ).astype(int)

        # ----------------------------------------------------
        # EDUCATION ONE-HOT
        # ----------------------------------------------------

        education_types = [
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Secondary / secondary special"
        ]

        if "NAME_EDUCATION_TYPE" in df.columns:

            for education_type in education_types:

                column_name = (
                    "NAME_EDUCATION_TYPE_"
                    + education_type
                )

                df[column_name] = (
                    df["NAME_EDUCATION_TYPE"]
                    == education_type
                ).astype(int)

        # ----------------------------------------------------
        # FAMILY STATUS ONE-HOT
        # ----------------------------------------------------

        if "NAME_FAMILY_STATUS" in df.columns:

            df[
                "NAME_FAMILY_STATUS_Married"
            ] = (
                df["NAME_FAMILY_STATUS"]
                == "Married"
            ).astype(int)

        # ====================================================
        # FINAL 40 FEATURES
        # ====================================================

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing required model features: "
                + str(missing_features)
            )

        # Keep ONLY model features
        df = df[self.feature_names]

        # Ensure numerical values
        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Check for missing values
        if df.isnull().any().any():

            missing = (
                df.columns[
                    df.isnull().any()
                ].tolist()
            )

            raise ValueError(
                "Missing/invalid values found in: "
                + str(missing)
            )

        return df