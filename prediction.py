# ============================================================
# PREDICTION CLASS
# ============================================================

import pandas as pd
import numpy as np


class Prediction:

    def __init__(self, model, feature_names):

        self.model = model
        self.feature_names = feature_names

    # --------------------------------------------------------
    # Validate new applicant data
    # --------------------------------------------------------

    def validate_input(self, X_new):

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in X_new.columns
        ]

        extra_features = [
            feature
            for feature in X_new.columns
            if feature not in self.feature_names
        ]

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )

        if extra_features:
            X_new = X_new.drop(
                columns=extra_features
            )

        # Ensure correct feature order
        X_new = X_new[
            self.feature_names
        ]

        return X_new

    # --------------------------------------------------------
    # Predict default probability
    # --------------------------------------------------------

    def predict_probability(self, X_new):

        X_new = self.validate_input(X_new)

        probability = self.model.predict_proba(
            X_new
        )[:, 1]

        return probability

    # --------------------------------------------------------
    # Predict class
    # --------------------------------------------------------

    def predict_class(self, X_new):

        X_new = self.validate_input(X_new)

        prediction = self.model.predict(
            X_new
        )

        return prediction

    # --------------------------------------------------------
    # Generate final prediction report
    # --------------------------------------------------------

    def predict(self, X_new):

        probability = self.predict_probability(
            X_new
        )[0]

        prediction = self.predict_class(
            X_new
        )[0]

        if prediction == 1:
            decision = "Default Risk"
        else:
            decision = "Non-Default Risk"

        return pd.DataFrame({
            "Default_Probability": [
                probability
            ],
            "Default_Probability_Percent": [
                probability * 100
            ],
            "Predicted_Class": [
                prediction
            ],
            "Risk_Decision": [
                decision
            ]
        })