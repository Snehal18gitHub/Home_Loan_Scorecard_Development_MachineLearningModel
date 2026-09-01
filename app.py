import streamlit as st
import pandas as pd
import numpy as np
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Home Loan Application Scorecard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# FLASK API CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:5000/predict"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 18px;
        color: #6B7280;
        margin-top: 0;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .result-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
    }

    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }

    .risk-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }

    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 Home Loan Application Scorecard</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Enter the applicant's details below to estimate the probability
    of home loan default.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

with st.container(border=True):

    st.subheader("👤 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Select Gender",
                "Female",
                "Male"
            ]
        )

    with col2:

        age = st.number_input(
            "Age (Years)",
            min_value=18,
            max_value=80,
            value=None,
            step=1,
            placeholder="Enter age"
        )

    with col3:

        employment_available = st.selectbox(
            "Employment Information Available?",
            [
                "Select Option",
                "Yes",
                "No"
            ]
        )


    # --------------------------------------------------------
    # CONDITIONAL EMPLOYMENT INPUT
    # --------------------------------------------------------

    employment_years = None

    if employment_available == "Yes":

        employment_years = st.number_input(
            "Employment Experience (Years)",
            min_value=0.0,
            max_value=50.0,
            value=None,
            step=0.5,
            placeholder="Enter employment experience"
        )

    elif employment_available == "No":

        st.info(
            "Employment information is unavailable. "
            "Employment experience will be treated as unavailable."
        )

        employment_years = 0.0


# ============================================================
# LOAN INFORMATION
# ============================================================

with st.container(border=True):

    st.subheader("💰 Loan Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=10000.0,
            max_value=10000000.0,
            value=None,
            step=10000.0,
            placeholder="Enter loan amount"
        )

    with col2:

        goods_price = st.number_input(
            "Property / Goods Price",
            min_value=10000.0,
            max_value=10000000.0,
            value=None,
            step=10000.0,
            placeholder="Enter property price"
        )

    with col3:

        annual_income = st.number_input(
            "Annual Income",
            min_value=10000.0,
            max_value=10000000.0,
            value=None,
            step=10000.0,
            placeholder="Enter annual income"
        )

    annuity = st.number_input(
        "Loan Annuity / Monthly EMI",
        min_value=100.0,
        max_value=1000000.0,
        value=None,
        step=500.0,
        placeholder="Enter monthly EMI"
    )


# ============================================================
# CREDIT INFORMATION
# ============================================================

with st.container(border=True):

    st.subheader("📊 Credit Information")

    ext_source_1_available = st.selectbox(
        "External Credit Score 1 Available?",
        [
            "Select Option",
            "Yes",
            "No"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        ext_source_2 = st.number_input(
            "External Credit Score 2",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder="Enter score between 0 and 1"
        )

    with col2:

        ext_source_3 = st.number_input(
            "External Credit Score 3",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder="Enter score between 0 and 1"
        )


    # --------------------------------------------------------
    # CONDITIONAL EXTERNAL SCORE 1
    # --------------------------------------------------------

    ext_source_1 = None

    if ext_source_1_available == "Yes":

        ext_source_1 = st.number_input(
            "External Credit Score 1",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder="Enter score between 0 and 1"
        )

    elif ext_source_1_available == "No":

        ext_source_1 = 0.0


# ============================================================
# EMPLOYMENT AND REGION
# ============================================================

with st.container(border=True):

    st.subheader("💼 Employment & Region")

    col1, col2, col3 = st.columns(3)

    with col1:

        flag_emp_phone = st.selectbox(
            "Employer Phone Available?",
            [
                "Select Option",
                "Yes",
                "No"
            ]
        )

    with col2:

        region_rating = st.selectbox(
            "Region Rating",
            [
                "Select Rating",
                1,
                2,
                3
            ]
        )

    with col3:

        region_rating_city = st.selectbox(
            "Region Rating in City",
            [
                "Select Rating",
                1,
                2,
                3
            ]
        )


# ============================================================
# PROPERTY INFORMATION
# ============================================================

with st.container(border=True):

    st.subheader("🏠 Property Information")

    col1, col2 = st.columns(2)

    with col1:

        apartments_avg = st.number_input(
            "Apartment Quality / Area Score",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder="Enter score between 0 and 1"
        )

    with col2:

        owns_vehicle = st.selectbox(
            "Owns Vehicle?",
            [
                "Select Option",
                "Yes",
                "No"
            ]
        )


# ============================================================
# FAMILY AND SOCIAL INFORMATION
# ============================================================

with st.container(border=True):

    st.subheader("👨‍👩‍👧 Family & Social Information")

    col1, col2 = st.columns(2)

    with col1:

        social_30 = st.number_input(
            "30-Day Social Circle Observations",
            min_value=0,
            max_value=100,
            value=None,
            step=1,
            placeholder="Enter value"
        )

    with col2:

        social_60 = st.number_input(
            "60-Day Social Circle Observations",
            min_value=0,
            max_value=100,
            value=None,
            step=1,
            placeholder="Enter value"
        )

    col1, col2 = st.columns(2)

    with col1:

        family_status = st.selectbox(
            "Family Status",
            [
                "Select Family Status",
                "Married",
                "Other"
            ]
        )


# ============================================================
# INCOME AND EDUCATION
# ============================================================

with st.container(border=True):

    st.subheader("🎓 Income & Education")

    col1, col2 = st.columns(2)

    with col1:

        income_type = st.selectbox(
            "Income Type",
            [
                "Select Income Type",
                "Commercial associate",
                "Pensioner",
                "State servant",
                "Working"
            ]
        )

    with col2:

        education = st.selectbox(
            "Education Level",
            [
                "Select Education",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Secondary / secondary special"
            ]
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

with st.container(border=True):

    st.subheader("📄 Document Information")

    document_options = [
        "Select Option",
        "No",
        "Yes"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        document_4 = st.selectbox(
            "Document 4",
            document_options
        )

    with col2:

        document_13 = st.selectbox(
            "Document 13",
            document_options
        )

    with col3:

        document_14 = st.selectbox(
            "Document 14",
            document_options
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        document_16 = st.selectbox(
            "Document 16",
            document_options
        )

    with col2:

        document_17 = st.selectbox(
            "Document 17",
            document_options
        )

    with col3:

        document_18 = st.selectbox(
            "Document 18",
            document_options
        )


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

with st.container(border=True):

    st.subheader("📋 Additional Information")

    days_id_publish = st.number_input(
        "Days Since ID Document Published",
        min_value=-30000,
        max_value=0,
        value=None,
        step=1,
        placeholder="Enter negative number of days"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Loan Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION PROCESS
# ============================================================

if predict_button:

    # ========================================================
    # REQUIRED INPUT VALIDATION
    # ========================================================

    required_values = {

        "Gender": gender,
        "Age": age,
        "Employment Information": employment_available,
        "Loan Amount": loan_amount,
        "Property / Goods Price": goods_price,
        "Annual Income": annual_income,
        "Loan Annuity / Monthly EMI": annuity,
        "External Credit Score 1 Availability": ext_source_1_available,
        "External Credit Score 2": ext_source_2,
        "External Credit Score 3": ext_source_3,
        "Employer Phone": flag_emp_phone,
        "Region Rating": region_rating,
        "Region Rating in City": region_rating_city,
        "Apartment Quality Score": apartments_avg,
        "Owns Vehicle": owns_vehicle,
        "30-Day Social Circle Observations": social_30,
        "60-Day Social Circle Observations": social_60,
        "Family Status": family_status,
        "Income Type": income_type,
        "Education": education,
        "Document 4": document_4,
        "Document 13": document_13,
        "Document 14": document_14,
        "Document 16": document_16,
        "Document 17": document_17,
        "Document 18": document_18,
        "Days Since ID Published": days_id_publish
    }


    # --------------------------------------------------------
    # Employment years required only if available
    # --------------------------------------------------------

    if employment_available == "Yes":

        required_values[
            "Employment Experience"
        ] = employment_years


    # --------------------------------------------------------
    # External score 1 required only if available
    # --------------------------------------------------------

    if ext_source_1_available == "Yes":

        required_values[
            "External Credit Score 1"
        ] = ext_source_1


    # --------------------------------------------------------
    # Find missing fields
    # --------------------------------------------------------

    missing_fields = []

    for field, value in required_values.items():

        if value is None:

            missing_fields.append(field)

        elif (
            isinstance(value, str)
            and value.startswith("Select")
        ):

            missing_fields.append(field)


    if missing_fields:

        st.error(
            "⚠️ Please complete all required fields before prediction."
        )

        with st.expander("View Missing Fields"):

            for field in missing_fields:

                st.write(f"• {field}")

        st.stop()


    # ========================================================
    # BASIC ENCODING
    # ========================================================

    gender_value = (
        0 if gender == "Female" else 1
    )

    emp_phone_value = (
        1 if flag_emp_phone == "Yes" else 0
    )

    employment_unavailable = (
        0 if employment_available == "Yes"
        else 1
    )

    vehicle_value = (
        1 if owns_vehicle == "Yes"
        else 0
    )


    document_4_value = (
        1 if document_4 == "Yes"
        else 0
    )

    document_13_value = (
        1 if document_13 == "Yes"
        else 0
    )

    document_14_value = (
        1 if document_14 == "Yes"
        else 0
    )

    document_16_value = (
        1 if document_16 == "Yes"
        else 0
    )

    document_17_value = (
        1 if document_17 == "Yes"
        else 0
    )

    document_18_value = (
        1 if document_18 == "Yes"
        else 0
    )


    # ========================================================
    # INCOME CATEGORY
    # ========================================================

    if annual_income <= 112500:

        income_category = 0

    elif annual_income <= 147150:

        income_category = 1

    elif annual_income <= 202500:

        income_category = 2

    else:

        income_category = 3


    # ========================================================
    # FINANCIAL RATIOS
    # ========================================================

    monthly_income = annual_income / 12

    emi_income_ratio = (
        annuity / monthly_income
        if monthly_income > 0
        else 0
    )

    ltv_ratio = (
        loan_amount / goods_price
        if goods_price > 0
        else 0
    )


    # ========================================================
    # EXTERNAL CREDIT FEATURES
    # ========================================================

    external_scores = [
        ext_source_1,
        ext_source_2,
        ext_source_3
    ]

    avg_external_score = np.mean(
        external_scores
    )

    max_external_score = np.max(
        external_scores
    )

    min_external_score = np.min(
        external_scores
    )

    ext_source_1_missing = (
        0
        if ext_source_1_available == "Yes"
        else 1
    )


    # ========================================================
    # DOCUMENT COMPLETENESS
    # ========================================================

    document_completeness_score = (
        document_4_value
        + document_13_value
        + document_14_value
        + document_16_value
        + document_17_value
        + document_18_value
    )


    # ========================================================
    # INCOME TYPE ENCODING
    # ========================================================

    income_commercial = (
        1
        if income_type == "Commercial associate"
        else 0
    )

    income_pensioner = (
        1
        if income_type == "Pensioner"
        else 0
    )

    income_state_servant = (
        1
        if income_type == "State servant"
        else 0
    )

    income_working = (
        1
        if income_type == "Working"
        else 0
    )


    # ========================================================
    # EDUCATION ENCODING
    # ========================================================

    education_higher = (
        1
        if education == "Higher education"
        else 0
    )

    education_incomplete = (
        1
        if education == "Incomplete higher"
        else 0
    )

    education_lower_secondary = (
        1
        if education == "Lower secondary"
        else 0
    )

    education_secondary = (
        1
        if education == "Secondary / secondary special"
        else 0
    )


    # ========================================================
    # FAMILY STATUS ENCODING
    # ========================================================

    family_married = (
        1
        if family_status == "Married"
        else 0
    )


    # ========================================================
    # CREATE FINAL 40 FEATURES
    # ========================================================

    new_applicant = pd.DataFrame([{

        "CODE_GENDER": gender_value,

        "AMT_CREDIT": loan_amount,

        "AMT_GOODS_PRICE": goods_price,

        "DAYS_ID_PUBLISH": days_id_publish,

        "FLAG_EMP_PHONE": emp_phone_value,

        "REGION_RATING_CLIENT": region_rating,

        "REGION_RATING_CLIENT_W_CITY":
            region_rating_city,

        "EXT_SOURCE_2": ext_source_2,

        "EXT_SOURCE_3": ext_source_3,

        "APARTMENTS_AVG": apartments_avg,

        "OBS_30_CNT_SOCIAL_CIRCLE":
            social_30,

        "OBS_60_CNT_SOCIAL_CIRCLE":
            social_60,

        "FLAG_DOCUMENT_4":
            document_4_value,

        "FLAG_DOCUMENT_13":
            document_13_value,

        "FLAG_DOCUMENT_14":
            document_14_value,

        "FLAG_DOCUMENT_16":
            document_16_value,

        "FLAG_DOCUMENT_17":
            document_17_value,

        "FLAG_DOCUMENT_18":
            document_18_value,

        "EXT_SOURCE_1_MISSING":
            ext_source_1_missing,

        "AGE_YEARS":
            age,

        "EMPLOYMENT_DATA_UNAVAILABLE":
            employment_unavailable,

        "EMPLOYMENT_YEARS":
            employment_years,

        "INCOME_CATEGORY":
            income_category,

        "EMI_INCOME_RATIO":
            emi_income_ratio,

        "LTV_RATIO":
            ltv_ratio,

        "OWNS_VEHICLE":
            vehicle_value,

        "AVG_EXTERNAL_SCORE":
            avg_external_score,

        "MAX_EXTERNAL_SCORE":
            max_external_score,

        "MIN_EXTERNAL_SCORE":
            min_external_score,

        "PROPERTY_QUALITY_SCORE":
            apartments_avg,

        "DOCUMENT_COMPLETENESS_SCORE":
            document_completeness_score,

        "NAME_INCOME_TYPE_Commercial associate":
            income_commercial,

        "NAME_INCOME_TYPE_Pensioner":
            income_pensioner,

        "NAME_INCOME_TYPE_State servant":
            income_state_servant,

        "NAME_INCOME_TYPE_Working":
            income_working,

        "NAME_EDUCATION_TYPE_Higher education":
            education_higher,

        "NAME_EDUCATION_TYPE_Incomplete higher":
            education_incomplete,

        "NAME_EDUCATION_TYPE_Lower secondary":
            education_lower_secondary,

        "NAME_EDUCATION_TYPE_Secondary / secondary special":
            education_secondary,

        "NAME_FAMILY_STATUS_Married":
            family_married

    }])


    # ========================================================
    # SEND DATA TO FLASK API
    # ========================================================

    try:

        # ----------------------------------------------------
        # Convert DataFrame to dictionary
        # ----------------------------------------------------

        payload = new_applicant.iloc[0].to_dict()


        # ----------------------------------------------------
        # Convert NumPy values to Python values
        # ----------------------------------------------------

        payload = {

            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )

            for key, value in payload.items()
        }


        # ----------------------------------------------------
        # Call Flask API
        # ----------------------------------------------------

        with st.spinner(
            "Connecting to Flask API and generating prediction..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )


        # ====================================================
        # SUCCESS RESPONSE
        # ====================================================

        if response.status_code == 200:

            result = response.json()

            prediction_id = result.get(
                "prediction_id"
            )

            probability_percent = float(
                result["default_probability_percent"]
            )

            predicted_class = int(
                result["predicted_class"]
            )

            risk_decision = str(
                result["risk_decision"]
            )

            risk_level = str(
                result["risk_level"]
            )


            # =================================================
            # DETERMINE UI RISK LEVEL
            # =================================================

            if risk_level == "Low Risk":

                risk_label = "🟢 LOW RISK"

                risk_class = "risk-low"

                risk_message = (
                    "The applicant has a relatively low "
                    "predicted probability of default."
                )

            elif risk_level == "Medium Risk":

                risk_label = "🟠 MODERATE RISK"

                risk_class = "risk-moderate"

                risk_message = (
                    "The applicant shows a moderate "
                    "predicted probability of default."
                )

            else:

                risk_label = "🔴 HIGH RISK"

                risk_class = "risk-high"

                risk_message = (
                    "The applicant has a high predicted "
                    "probability of default."
                )


            # =================================================
            # DISPLAY SUCCESS MESSAGE
            # =================================================

            st.success(
                "✅ Prediction generated and stored successfully!"
            )


            # =================================================
            # LOAN RISK ASSESSMENT
            # =================================================

            st.divider()

            st.header(
                "📊 Loan Risk Assessment"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Prediction ID",
                    str(prediction_id)
                )


            with col2:

                st.metric(
                    "Default Probability",
                    f"{probability_percent:.2f}%"
                )


            with col3:

                st.metric(
                    "Predicted Class",
                    str(predicted_class)
                )


            with col4:

                st.metric(
                    "Model Decision",
                    risk_decision
                )


            # =================================================
            # DEFAULT PROBABILITY BAR
            # =================================================

            st.subheader(
                "Default Risk Probability"
            )

            probability_value = max(
                0,
                min(
                    int(probability_percent),
                    100
                )
            )

            st.progress(
                probability_value
            )


            # =================================================
            # RISK CARD
            # =================================================

            st.markdown(
                f"""
                <div class="result-card {risk_class}">
                    <h2>{risk_label}</h2>
                    <p>{risk_message}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # PREDICTION DETAILS
            # =================================================

            with st.expander(
                "📋 View Prediction Details"
            ):

                result_display = pd.DataFrame({

                    "Metric": [

                        "Prediction ID",

                        "Default Probability",

                        "Predicted Class",

                        "Model Risk Decision",

                        "Risk Level"

                    ],

                    "Result": [

                        str(prediction_id),

                        f"{probability_percent:.2f}%",

                        str(predicted_class),

                        risk_decision,

                        risk_level

                    ]

                })

                st.table(
                    result_display
                )


            # =================================================
            # MODEL INPUT
            # =================================================

            with st.expander(
                "🔍 View Model Input (40 Features)"
            ):

                st.dataframe(
                    new_applicant.T.rename(
                        columns={0: "Value"}
                    ),
                    use_container_width=True
                )


        # ====================================================
        # FLASK API ERROR
        # ====================================================

        else:

            try:

                error_data = response.json()

                error_message = error_data.get(
                    "message",
                    "Prediction failed"
                )

            except Exception:

                error_message = response.text


            st.error(
                f"❌ Flask API Error: {error_message}"
            )


            # ------------------------------------------------
            # Show missing features if returned by API
            # ------------------------------------------------

            if isinstance(
                error_data if "error_data" in locals()
                else None,
                dict
            ):

                missing_features = error_data.get(
                    "missing_features"
                )

                if missing_features:

                    with st.expander(
                        "View Missing API Features"
                    ):

                        for feature in missing_features:

                            st.write(
                                f"• {feature}"
                            )


    # ========================================================
    # FLASK CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Unable to connect to Flask API."
        )

        st.info(
            "Please make sure flask_api.py is running "
            "on http://127.0.0.1:5000"
        )


    # ========================================================
    # REQUEST TIMEOUT
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ Flask API request timed out."
        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Prediction request failed."
        )

        st.exception(e)

