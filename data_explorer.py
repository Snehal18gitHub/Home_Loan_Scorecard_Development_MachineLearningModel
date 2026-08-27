import pandas as pd
import numpy as np

# ============================================================
# DATA EXPLORER
# ============================================================

class DataExplorer:

    def __init__(self, df):
        self.df = df

    def basic_info(self):

        print("=" * 60)
        print("BASIC DATA INFORMATION")
        print("=" * 60)

        print("\nDataset Shape:")
        print(self.df.shape)

        print("\nData Types:")
        print(self.df.dtypes)

        print("\nMissing Values:")
        print(self.df.isnull().sum().sort_values(ascending=False))

        return self

    def target_analysis(self, target_column):

        print("\nTarget Distribution:")
        print(self.df[target_column].value_counts())

        print("\nTarget Percentage:")
        print(
            self.df[target_column]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        return self

    def missing_value_analysis(self):

        missing_percentage = (
            self.df.isnull()
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        return missing_percentage

    def numerical_analysis(self):

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        return self.df[numerical_columns].describe()

    def categorical_analysis(self):

        categorical_columns = self.df.select_dtypes(
            include="object"
        ).columns

        for column in categorical_columns:
            print(f"\n{'=' * 60}")
            print(column)
            print("=" * 60)

            print(self.df[column].value_counts(dropna=False))

        return self

    def correlation_analysis(self):

        numerical_df = self.df.select_dtypes(include=np.number)

        return numerical_df.corr()