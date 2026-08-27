import pandas as pd
import numpy as np


# ============================================================
# FEATURE ENGINEER
# ============================================================

class FeatureEngineer:
    """
    Creates all engineered features for the Home Loan
    Scorecard project.

    The original input DataFrame is not modified.
    Feature engineering is performed on a safe copy.
    """

    def __init__(self, df):
        self.df = df.copy()

    # --------------------------------------------------------
    # 2.1 AGE FEATURES
    # --------------------------------------------------------
    def create_age_features(self):

        self.df["AGE_YEARS"] = (
            abs(self.df["DAYS_BIRTH"]) / 365
        ).astype(int)

        conditions = [
            self.df["AGE_YEARS"] <= 30,
            self.df["AGE_YEARS"] <= 45,
            self.df["AGE_YEARS"] <= 60,
            self.df["AGE_YEARS"] > 60
        ]

        choices = [
            "Young",
            "Adult",
            "Middle Age",
            "Senior"
        ]

        self.df["AGE_GROUP"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.2 EMPLOYMENT EXPERIENCE FEATURES
    # --------------------------------------------------------
    def create_employment_features(self):

        self.df["EMPLOYMENT_DATA_UNAVAILABLE"] = np.where(
            self.df["DAYS_EMPLOYED"] == 365243,
            1,
            0
        )

        self.df["EMPLOYMENT_YEARS"] = np.where(
            self.df["DAYS_EMPLOYED"] == 365243,
            0,
            abs(self.df["DAYS_EMPLOYED"]) / 365
        )

        conditions = [
            self.df["EMPLOYMENT_DATA_UNAVAILABLE"] == 1,
            self.df["EMPLOYMENT_YEARS"] < 2,
            self.df["EMPLOYMENT_YEARS"].between(2, 10),
            self.df["EMPLOYMENT_YEARS"] > 10
        ]

        choices = [
            "Data Unavailable",
            "New Employee",
            "Experienced Employee",
            "Highly Experienced Employee"
        ]

        self.df["EMPLOYMENT_STABILITY"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.3 INCOME CATEGORY
    # --------------------------------------------------------
    def create_income_features(self):

        conditions = [
            self.df["AMT_INCOME_TOTAL"] <= 112500,
            self.df["AMT_INCOME_TOTAL"].between(
                112500, 147150
            ),
            self.df["AMT_INCOME_TOTAL"].between(
                147150, 202500
            ),
            self.df["AMT_INCOME_TOTAL"] > 202500
        ]

        choices = [
            "Low Income",
            "Medium Income",
            "High Income",
            "Very High Income"
        ]

        self.df["INCOME_CATEGORY"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.4 TO 2.7 INCOME AND LOAN FEATURES
    # --------------------------------------------------------
    def create_loan_features(self):

        self.df["LTI_RATIO"] = (
            self.df["AMT_CREDIT"]
            / self.df["AMT_INCOME_TOTAL"]
        )

        self.df["EMI_INCOME_RATIO"] = (
            self.df["AMT_ANNUITY"]
            / (self.df["AMT_INCOME_TOTAL"] / 12)
        )

        self.df["LTV_RATIO"] = (
            self.df["AMT_CREDIT"]
            / self.df["AMT_GOODS_PRICE"]
        )

        self.df["DISPOSABLE_MONTHLY_INCOME"] = (
            (self.df["AMT_INCOME_TOTAL"] / 12)
            - self.df["AMT_ANNUITY"]
        )

        return self

    # --------------------------------------------------------
    # 2.8 FAMILY SIZE CATEGORY
    # --------------------------------------------------------
    def create_family_size_feature(self):

        conditions = [
            self.df["CNT_FAM_MEMBERS"].between(1, 3),
            self.df["CNT_FAM_MEMBERS"].between(4, 5),
            self.df["CNT_FAM_MEMBERS"] >= 6
        ]

        choices = [
            "Small Family",
            "Medium Family",
            "Large Family"
        ]

        self.df["FAMILY_SIZE_CATEGORY"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.9 CHILDREN CATEGORY
    # --------------------------------------------------------
    def create_children_feature(self):

        conditions = [
            self.df["CNT_CHILDREN"] == 0,
            self.df["CNT_CHILDREN"] == 1,
            self.df["CNT_CHILDREN"] >= 2
        ]

        choices = [
            "No Children",
            "One Child",
            "Two or More Children"
        ]

        self.df["CHILDREN_CATEGORY"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.10 RESIDENTIAL STABILITY
    # --------------------------------------------------------
    def create_residential_features(self):

        self.df["REGISTRATION_YEARS"] = (
            abs(self.df["DAYS_REGISTRATION"]) / 365
        )

        conditions = [
            (
                (self.df["REGISTRATION_YEARS"] >= 10)
                & (self.df["FLAG_OWN_REALTY"] == "Y")
                & (
                    self.df["NAME_HOUSING_TYPE"]
                    == "House / apartment"
                )
            ),
            (
                (self.df["REGISTRATION_YEARS"] >= 5)
                & (
                    (self.df["FLAG_OWN_REALTY"] == "Y")
                    | (
                        self.df["NAME_HOUSING_TYPE"]
                        == "House / apartment"
                    )
                )
            ),
            (
                (self.df["REGISTRATION_YEARS"] < 5)
                | (self.df["FLAG_OWN_REALTY"] == "N")
                | (
                    self.df["NAME_HOUSING_TYPE"].isin([
                        "With parents",
                        "Rented apartment"
                    ])
                )
            )
        ]

        choices = [
            "Stable",
            "Moderately Stable",
            "Unstable"
        ]

        self.df["RESIDENTIAL_STABILITY"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.11 VEHICLE OWNERSHIP
    # --------------------------------------------------------
    def create_vehicle_ownership_feature(self):

        conditions = [
            self.df["FLAG_OWN_CAR"] == "Y",
            self.df["FLAG_OWN_CAR"] == "N"
        ]

        choices = [
            "Yes",
            "No"
        ]

        self.df["OWNS_VEHICLE"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.12 PROPERTY OWNERSHIP
    # --------------------------------------------------------
    def create_property_ownership_feature(self):

        conditions = [
            self.df["FLAG_OWN_REALTY"] == "Y",
            self.df["FLAG_OWN_REALTY"] == "N"
        ]

        choices = [
            "Yes",
            "No"
        ]

        self.df["OWNS_PROPERTY"] = np.select(
            conditions,
            choices,
            default="Unknown"
        )

        return self

    # --------------------------------------------------------
    # 2.13 BUREAU INQUIRY FEATURES
    # --------------------------------------------------------
    def create_bureau_inquiry_features(self):

        self.df["TOTAL_BUREAU_INQUIRIES"] = (
            self.df["AMT_REQ_CREDIT_BUREAU_YEAR"]
        )

        self.df["RECENT_BUREAU_INQUIRIES"] = (
            self.df["AMT_REQ_CREDIT_BUREAU_MON"]
        )

        return self

    # --------------------------------------------------------
    # 2.14 EXTERNAL CREDIT SCORE FEATURES
    # --------------------------------------------------------
    def create_external_credit_features(self):

        external_cols = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        ]

        self.df["AVG_EXTERNAL_SCORE"] = (
            self.df[external_cols].mean(axis=1)
        )

        self.df["MAX_EXTERNAL_SCORE"] = (
            self.df[external_cols].max(axis=1)
        )

        self.df["MIN_EXTERNAL_SCORE"] = (
            self.df[external_cols].min(axis=1)
        )

        return self

    # --------------------------------------------------------
    # 2.15 PROPERTY QUALITY SCORE
    # --------------------------------------------------------
    def create_property_quality_feature(self):

        property_cols = [
            "APARTMENTS_AVG",
            "BASEMENTAREA_AVG",
            "LIVINGAREA_AVG",
            "YEARS_BUILD_AVG",
            "FLOORSMAX_AVG",
            "ELEVATORS_AVG"
        ]

        self.df["PROPERTY_QUALITY_SCORE"] = (
            self.df[property_cols].mean(axis=1)
        )

        return self

    # --------------------------------------------------------
    # 2.16 CONTACT VERIFICATION SCORE
    # --------------------------------------------------------
    def create_contact_verification_feature(self):

        contact_cols = [
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE"
        ]

        self.df["CONTACT_VERIFICATION_SCORE"] = (
            self.df[contact_cols].sum(axis=1)
        )

        return self

    # --------------------------------------------------------
    # 2.17 DOCUMENT COMPLETENESS SCORE
    # --------------------------------------------------------
    def create_document_completeness_feature(self):

        document_cols = [
            col for col in self.df.columns
            if col.startswith("FLAG_DOCUMENT")
        ]

        self.df["DOCUMENT_COMPLETENESS_SCORE"] = (
            self.df[document_cols].sum(axis=1)
        )

        return self

    # --------------------------------------------------------
    # 2.18 SOCIAL RISK SCORE
    # --------------------------------------------------------
    def create_social_risk_feature(self):

        self.df["SOCIAL_RISK_SCORE"] = (
            self.df["OBS_30_CNT_SOCIAL_CIRCLE"]
            + 2 * self.df["DEF_30_CNT_SOCIAL_CIRCLE"]
            + self.df["OBS_60_CNT_SOCIAL_CIRCLE"]
            + 2 * self.df["DEF_60_CNT_SOCIAL_CIRCLE"]
        )

        return self

    # --------------------------------------------------------
    # GET ENGINEERED DATA
    # --------------------------------------------------------
    def get_data(self):

        return self.df.copy()