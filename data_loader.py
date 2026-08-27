import pandas as pd


# ============================================================
# DATA LOADER
# ============================================================

class DataLoader:
    """
    Responsible for loading the dataset.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):

        self.df = pd.read_csv(self.file_path)

        print("=" * 60)
        print("DATA LOADED SUCCESSFULLY")
        print("=" * 60)
        print(f"Dataset Shape: {self.df.shape}")

        return self.df