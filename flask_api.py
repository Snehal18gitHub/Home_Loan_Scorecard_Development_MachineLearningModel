from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import mysql.connector

from prediction import Prediction


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = "models/best_model.pkl"
FEATURE_PATH = "models/feature_names.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    predictor = Prediction(
        model=model,
        feature_names=feature_names
    )

    print("Model loaded successfully")
    print("Model type:", type(model).__name__)
    print("Number of features:", len(feature_names))

except Exception as e:
    print("Error loading model:", e)

    model = None
    feature_names = None
    predictor = None


# ============================================================
# MYSQL CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Shivanya@123",
    "database": "home_loan_db"
}


# ============================================================
# EXACT 40 MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "CODE_GENDER",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "DAYS_ID_PUBLISH",
    "FLAG_EMP_PHONE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "EXT_SOURCE_1_MISSING",
    "AGE_YEARS",
    "EMPLOYMENT_DATA_UNAVAILABLE",
    "EMPLOYMENT_YEARS",
    "INCOME_CATEGORY",
    "EMI_INCOME_RATIO",
    "LTV_RATIO",
    "OWNS_VEHICLE",
    "AVG_EXTERNAL_SCORE",
    "MAX_EXTERNAL_SCORE",
    "MIN_EXTERNAL_SCORE",
    "PROPERTY_QUALITY_SCORE",
    "DOCUMENT_COMPLETENESS_SCORE",
    "NAME_INCOME_TYPE_Commercial associate",
    "NAME_INCOME_TYPE_Pensioner",
    "NAME_INCOME_TYPE_State servant",
    "NAME_INCOME_TYPE_Working",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Incomplete higher",
    "NAME_EDUCATION_TYPE_Lower secondary",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_FAMILY_STATUS_Married"
]


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )


# ============================================================
# HOME API
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Home Loan Scorecard API is running",
        "model_loaded": model is not None,
        "number_of_features": len(MODEL_FEATURES)
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded"
        }), 500

    # --------------------------------------------------------
    # Read JSON request
    # --------------------------------------------------------

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Request body must be a JSON object"
        }), 400

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = []

    for feature in MODEL_FEATURES:
        if feature not in data:
            missing_features.append(feature)

    if missing_features:
        return jsonify({
            "status": "error",
            "message": "Required features are missing",
            "missing_features": missing_features
        }), 400

    # --------------------------------------------------------
    # Create DataFrame and Generate Prediction
    # --------------------------------------------------------

    try:
        X_new = pd.DataFrame(
            [data],
            columns=MODEL_FEATURES
        )

        result = predictor.predict(X_new)

        # ----------------------------------------------------
        # Extract results
        # ----------------------------------------------------

        probability = float(
            result["Default_Probability"].iloc[0]
        )

        probability_percent = float(
            result["Default_Probability_Percent"].iloc[0]
        )

        predicted_class = int(
            result["Predicted_Class"].iloc[0]
        )

        risk_decision = str(
            result["Risk_Decision"].iloc[0]
        )

        # ----------------------------------------------------
        # Determine risk level
        # ----------------------------------------------------

        if probability < 0.30:
            risk_level = "Low Risk"

        elif probability < 0.60:
            risk_level = "Medium Risk"

        else:
            risk_level = "High Risk"

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "error": str(e)
        }), 500

    # ========================================================
    # SAVE PREDICTION TO MYSQL
    # ========================================================

    connection = None
    cursor = None
    prediction_id = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # ----------------------------------------------------
        # Columns to insert
        # ----------------------------------------------------

        columns = (
            MODEL_FEATURES
            + [
                "prediction",
                "risk_decision",
                "risk_level",
                "probability",
                "probability_percent"
            ]
        )

        # ----------------------------------------------------
        # Create SQL column string
        # ----------------------------------------------------

        column_names = ", ".join(
            "`" + column + "`"
            for column in columns
        )

        # ----------------------------------------------------
        # Create placeholders
        # ----------------------------------------------------

        placeholders = ", ".join(
            ["%s"] * len(columns)
        )

        # ----------------------------------------------------
        # INSERT query
        # ----------------------------------------------------

        insert_query = (
            "INSERT INTO predictions ("
            + column_names
            + ") VALUES ("
            + placeholders
            + ")"
        )

        # ----------------------------------------------------
        # Prepare 40 input values
        # ----------------------------------------------------

        input_values = []

        for feature in MODEL_FEATURES:
            input_values.append(data[feature])

        # ----------------------------------------------------
        # Add prediction output values
        # ----------------------------------------------------

        output_values = [
            predicted_class,
            risk_decision,
            risk_level,
            probability,
            probability_percent
        ]

        # ----------------------------------------------------
        # Combine all values
        # ----------------------------------------------------

        values = tuple(
            input_values + output_values
        )

        # ----------------------------------------------------
        # Execute INSERT
        # ----------------------------------------------------

        cursor.execute(
            insert_query,
            values
        )

        connection.commit()

        # ----------------------------------------------------
        # Get generated prediction ID
        # ----------------------------------------------------

        prediction_id = cursor.lastrowid

    except mysql.connector.Error as e:

        if connection is not None:
            connection.rollback()

        return jsonify({
            "status": "error",
            "message": "Database insertion failed",
            "database_error": str(e)
        }), 500

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return jsonify({
        "status": "success",
        "prediction_id": prediction_id,
        "predicted_class": predicted_class,
        "risk_decision": risk_decision,
        "risk_level": risk_level,
        "default_probability": round(
            probability,
            4
        ),
        "default_probability_percent": round(
            probability_percent,
            2
        ),
        "message": "Prediction generated and stored successfully"
    }), 200


# ============================================================
# GET ALL PREDICTIONS
# ============================================================

@app.route("/predictions", methods=["GET"])
def get_predictions():

    connection = None
    cursor = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            "SELECT * FROM predictions "
            "ORDER BY prediction_date DESC"
        )

        records = cursor.fetchall()

        # ----------------------------------------------------
        # Convert datetime to JSON-compatible string
        # ----------------------------------------------------

        for record in records:

            if record["prediction_date"] is not None:
                record["prediction_date"] = (
                    record["prediction_date"].isoformat()
                )

        return jsonify({
            "status": "success",
            "count": len(records),
            "predictions": records
        }), 200

    except mysql.connector.Error as e:

        return jsonify({
            "status": "error",
            "message": "Unable to retrieve predictions",
            "database_error": str(e)
        }), 500

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ============================================================
# GET SINGLE PREDICTION
# ============================================================

@app.route(
    "/predictions/<int:prediction_id>",
    methods=["GET"]
)
def get_prediction(prediction_id):

    connection = None
    cursor = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            "SELECT * FROM predictions "
            "WHERE prediction_id = %s",
            (prediction_id,)
        )

        record = cursor.fetchone()

        if record is None:
            return jsonify({
                "status": "error",
                "message": "Prediction not found"
            }), 404

        # ----------------------------------------------------
        # Convert datetime to string
        # ----------------------------------------------------

        if record["prediction_date"] is not None:
            record["prediction_date"] = (
                record["prediction_date"].isoformat()
            )

        return jsonify({
            "status": "success",
            "prediction": record
        }), 200

    except mysql.connector.Error as e:

        return jsonify({
            "status": "error",
            "message": "Unable to retrieve prediction",
            "database_error": str(e)
        }), 500

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

