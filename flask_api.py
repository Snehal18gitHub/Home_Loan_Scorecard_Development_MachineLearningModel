# ============================================================
# HOME LOAN SCORECARD - FLASK API
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import joblib
import mysql.connector
import os

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# MODEL FILE PATHS
# ============================================================

MODEL_PATH = os.path.join("models", "best_model.pkl")
FEATURE_PATH = os.path.join("models", "feature_names.pkl")

# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    print("============================================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("============================================================")
    print("Model Type       :", type(model).__name__)
    print("Number Features  :", len(feature_names))
    print("============================================================")

except Exception as e:

    print("ERROR LOADING MODEL:")
    print(e)

    model = None
    feature_names = []


# ============================================================
# MYSQL CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_MYSQL_PASSWORD",
    "database":  "home_loan_db"
}


# ============================================================
# MYSQL CONNECTION
# ============================================================

def get_db_connection():

    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )


# ============================================================
# BINARY ENCODING
# ============================================================

def encode_binary_features(df):

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

        if col in df.columns:

            df[col] = df[col].map(mapping)

    return df


# ============================================================
# GENDER ENCODING
# ============================================================

def encode_gender(df):

    if "CODE_GENDER" in df.columns:

        gender_mapping = {
            "F": 0,
            "M": 1,
            "XNA": -1
        }

        df["CODE_GENDER"] = (
            df["CODE_GENDER"]
            .map(gender_mapping)
        )

    return df


# ============================================================
# ORDINAL ENCODING
# ============================================================

def encode_ordinal_features(df):

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

        if col in df.columns:

            df[col] = df[col].map(mapping)

    return df


# ============================================================
# FREQUENCY ENCODING
# ============================================================
#
# IMPORTANT:
# The frequency values below MUST be the same values used
# during model training.
#
# If your training pipeline used the original training data
# to calculate frequency encoding, those mappings should
# ideally be saved as artifacts.
#
# For now, this function expects saved mappings.
# ============================================================

FREQUENCY_MAP_PATH = os.path.join(
    "models",
    "frequency_maps.pkl"
)


def load_frequency_maps():

    if os.path.exists(FREQUENCY_MAP_PATH):

        return joblib.load(FREQUENCY_MAP_PATH)

    return {}


frequency_maps = load_frequency_maps()


def encode_frequency_features(df):

    frequency_cols = [
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE"
    ]

    for col in frequency_cols:

        if col in df.columns:

            if col in frequency_maps:

                df[col] = (
                    df[col]
                    .map(frequency_maps[col])
                    .fillna(0)
                )

            else:

                raise ValueError(
                    f"Frequency mapping for {col} "
                    "is not available."
                )

    return df


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def encode_onehot_features(df):

    onehot_cols = [

        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "WEEKDAY_APPR_PROCESS_START"
    ]

    existing_cols = [
        col
        for col in onehot_cols
        if col in df.columns
    ]

    if existing_cols:

        df = pd.get_dummies(
            df,
            columns=existing_cols,
            drop_first=True,
            dtype=int
        )

    return df


# ============================================================
# EMPLOYMENT STABILITY ENCODING
# ============================================================

def encode_employment_stability(df):

    if "EMPLOYMENT_STABILITY" in df.columns:

        df = pd.get_dummies(
            df,
            columns=["EMPLOYMENT_STABILITY"],
            drop_first=True,
            dtype=int
        )

    return df


# ============================================================
# ALIGN FEATURES WITH TRAINING FEATURES
# ============================================================

def align_features(df):

    # --------------------------------------------------------
    # Add missing model features
    # --------------------------------------------------------

    for feature in feature_names:

        if feature not in df.columns:

            df[feature] = 0


    # --------------------------------------------------------
    # Remove unwanted columns
    # --------------------------------------------------------

    df = df[feature_names]


    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    for col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Handle remaining missing values
    # --------------------------------------------------------

    df = df.fillna(0)


    return df


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_input(data):

    df = pd.DataFrame([data])

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    df = df.loc[:, feature_names]
    df = df.apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    )

    invalid_features = df.columns[df.isna().any()].tolist()

    if invalid_features:
        raise ValueError(
            "Feature values must be numeric and non-null. "
            "Invalid features: "
            + ", ".join(invalid_features)
        )

    if df.shape != (1, len(feature_names)):
        raise ValueError(
            "Expected exactly 1 row and "
            f"{len(feature_names)} model features; received "
            f"{df.shape[0]} rows and {df.shape[1]} features."
        )

    return df


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(data):

    if not isinstance(data, dict):

        return False, "Input must be a JSON object."

    if len(data) == 0:

        return False, "Input data cannot be empty."

    return True, None


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "application":
            "Home Loan Application Scorecard",

        "status":
            "API is running successfully",

        "model":
            type(model).__name__ if model else None,

        "features":
            len(feature_names)

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "healthy",

        "model_loaded":
            model is not None,

        "feature_count":
            len(feature_names)

    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "Model is not loaded."

            }), 500


        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        data = request.get_json()

        print("\n============================================================")
        print("PREDICTION REQUEST RECEIVED")
        print("============================================================")


        # ----------------------------------------------------
        # Validate JSON
        # ----------------------------------------------------

        is_valid, error_message = validate_input(data)

        if not is_valid:

            return jsonify({

                "success": False,

                "error": error_message

            }), 400


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        try:
            X_new = preprocess_input(data)
        except ValueError as input_error:
            return jsonify({
                "success": False,
                "error": str(input_error)
            }), 400


        # ----------------------------------------------------
        # Final safety check
        # ----------------------------------------------------

        if X_new.shape[1] != len(feature_names):

            return jsonify({

                "success": False,

                "error":
                    "Feature count mismatch.",

                "expected_features":
                    len(feature_names),

                "received_features":
                    X_new.shape[1]

            }), 500


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(X_new)[0]


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        classes = list(model.classes_)
        class_one_index = next(
            (
                index
                for index, class_label in enumerate(classes)
                if class_label == 1
            ),
            None
        )

        if class_one_index is None:
            raise ValueError(
                "Model does not contain class 1; "
                "cannot calculate default probability."
            )

        probability = float(
            model.predict_proba(X_new)[0][class_one_index]
        )


        # ----------------------------------------------------
        # Risk calculation
        # ----------------------------------------------------

        risk_score = round(
            probability * 100,
            2
        )

        if risk_score < 30:

            risk_level = "Low Risk"

        elif risk_score < 60:

            risk_level = "Medium Risk"

        else:

            risk_level = "High Risk"


        # ----------------------------------------------------
        # Application decision
        # ----------------------------------------------------

        if int(prediction) == 0:

            decision = "Likely Approved"

        else:

            decision = "High Default Risk"


        # ----------------------------------------------------
        # Store prediction in MySQL
        # ----------------------------------------------------

        prediction_id = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            insert_columns = list(feature_names) + [
                "prediction",
                "risk_decision",
                "risk_level",
                "probability",
                "probability_percent"
            ]
            quoted_columns = ", ".join(
                f"`{column.replace('`', '``')}`"
                for column in insert_columns
            )
            placeholders = ", ".join(
                ["%s"] * len(insert_columns)
            )

            query = (
                "INSERT INTO predictions "
                f"({quoted_columns}) "
                f"VALUES ({placeholders})"
            )

            values = tuple(X_new.iloc[0].tolist()) + (
                int(prediction),
                decision,
                risk_level,
                probability,
                risk_score
            )

            cursor.execute(
                query,
                values
            )

            prediction_id = cursor.lastrowid

            connection.commit()

            cursor.close()
            connection.close()

            database_status = "Saved successfully"


        except Exception as db_error:

            error_code = getattr(db_error, "errno", None)
            error_message = getattr(db_error, "msg", str(db_error))
            sql_state = getattr(db_error, "sqlstate", None)

            print("MYSQL DATABASE ERROR")
            print("Error type:", type(db_error).__name__)
            print("Error code:", error_code)
            print("SQL state:", sql_state)
            print("Error message:", error_message)
            print("Exception:", repr(db_error))

            if "query" in locals():
                print("SQL statement:")
                print(query)

            database_status = (
                "Prediction generated, "
                "but database storage failed. "
                f"MySQL error code: {error_code}; "
                f"SQL state: {sql_state}; "
                f"message: {error_message}"
            )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        response = {

            "success": True,

            "prediction_id": prediction_id,

            "predicted_class": int(prediction),

            "default_probability":
                round(
                    probability,
                    4
                ),

            "default_probability_percent":
                risk_score,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "risk_category": 
                risk_level,

            "risk_decision":
                decision,

            "decision":
                decision,

            "database":
                database_status

        }


        print("\n============================================================")
        print("PREDICTION RESULT")
        print("============================================================")

        print(response)


        return jsonify(response), 200


    except Exception as e:

        print("\n============================================================")
        print("PREDICTION ERROR")
        print("============================================================")

        print(str(e))


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# GET ALL PREDICTIONS
# ============================================================

@app.route("/predictions", methods=["GET"])
def get_predictions():

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute("""
            SELECT *
            FROM loan_predictions
            ORDER BY id DESC
        """)

        records = cursor.fetchall()

        cursor.close()
        connection.close()

        return jsonify({

            "success": True,

            "count": len(records),

            "data": records

        }), 200


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )