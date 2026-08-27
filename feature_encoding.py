import pandas as pd


# ============================================================
# FEATURE ENCODER
# ============================================================

class FeatureEncoder:
    """
    Handles feature encoding for the Home Loan Scorecard.

    The original DataFrame is not modified.
    All encoding is performed on a safe copy.
    """

    def __init__(self, df):
        self.df = df.copy()

    # --------------------------------------------------------
    # STEP 5.1: BINARY VARIABLE ENCODING
    # --------------------------------------------------------
    def encode_binary_features(self):

        binary_mapping = {

            "NAME_CONTRACT_TYPE": {
                "Cash loans": 0,
                "Revolving loans": 1
            },

            "FLAG_OWN_CAR": {
                "N": 0,
                "Y": 1
            },

            "FLAG_OWN_REALTY": {
                "N": 0,
                "Y": 1
            },

            "OWNS_VEHICLE": {
                "No": 0,
                "Yes": 1
            },

            "OWNS_PROPERTY": {
                "No": 0,
                "Yes": 1
            }
        }

        for col, mapping in binary_mapping.items():

            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)

        print("Binary Encoding Completed")

        return self

    # --------------------------------------------------------
    # STEP 5.2: ONE-HOT ENCODING
    # --------------------------------------------------------
    def encode_onehot_features(self):

        onehot_cols = [
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "WEEKDAY_APPR_PROCESS_START"
        ]

        existing_cols = [
            col for col in onehot_cols
            if col in self.df.columns
        ]

        self.df = pd.get_dummies(
            self.df,
            columns=existing_cols,
            drop_first=True,
            dtype=int
        )

        print("One-Hot Encoding Completed")

        return self

    # --------------------------------------------------------
    # STEP 5.3: CODE_GENDER ENCODING
    # --------------------------------------------------------
    def encode_gender(self):

        if "CODE_GENDER" in self.df.columns:

            self.df["CODE_GENDER"] = (
                self.df["CODE_GENDER"].map({
                    "F": 0,
                    "M": 1,
                    "XNA": -1
                })
            )

        print("CODE_GENDER Encoding Completed")

        return self

    # --------------------------------------------------------
    # STEP 5.4: ORDINAL ENCODING
    # --------------------------------------------------------
    def encode_ordinal_features(self):

        ordinal_mappings = {

            "AGE_GROUP": {
                "Young": 0,
                "Adult": 1,
                "Middle Age": 2,
                "Senior": 3
            },

            "INCOME_CATEGORY": {
                "Low Income": 0,
                "Medium Income": 1,
                "High Income": 2,
                "Very High Income": 3
            },

            "FAMILY_SIZE_CATEGORY": {
                "Small Family": 0,
                "Medium Family": 1,
                "Large Family": 2
            },

            "CHILDREN_CATEGORY": {
                "No Children": 0,
                "One Child": 1,
                "Two or More Children": 2
            },

            "RESIDENTIAL_STABILITY": {
                "Unstable": 0,
                "Moderately Stable": 1,
                "Stable": 2
            }
        }

        for col, mapping in ordinal_mappings.items():

            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)

        print("Ordinal Encoding Completed")

        return self

    # --------------------------------------------------------
    # STEP 5.5: EMPLOYMENT STABILITY ENCODING
    # --------------------------------------------------------
    def encode_employment_stability(self):

        if "EMPLOYMENT_STABILITY" in self.df.columns:

            self.df = pd.get_dummies(
                self.df,
                columns=["EMPLOYMENT_STABILITY"],
                drop_first=True,
                dtype=int
            )

        print("Employment Stability Encoding Completed")

        return self

    # --------------------------------------------------------
    # STEP 5.6: FREQUENCY ENCODING
    # --------------------------------------------------------
    def encode_frequency_features(self):

        frequency_cols = [
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE"
        ]

        for col in frequency_cols:

            if col in self.df.columns:

                frequency_map = (
                    self.df[col]
                    .value_counts(normalize=True)
                )

                self.df[col] = (
                    self.df[col]
                    .map(frequency_map)
                )

        print("Frequency Encoding Completed")

        return self

    # --------------------------------------------------------
    # GET ENCODED DATA
    # --------------------------------------------------------
    def get_data(self):

        return self.df.copy()