from config import DATA_PATH, TARGET_COLUMN

from data_loader import DataLoader
from data_explorer import DataExplorer
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from feature_encoding import FeatureEncoder
from feature_scaling import FeatureScaler
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from model_cross_validation import ModelCrossValidator
from model_evaluator import ModelEvaluator
from shap_analyzer import SHAPAnalyzer

from sklearn.model_selection import train_test_split


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

loader = DataLoader(DATA_PATH)

df = loader.load_data()


# ============================================================
# STEP 2: DATA EXPLORATION
# ============================================================

explorer = DataExplorer(df)

# 1. Basic Information
explorer.basic_info()

# 2. Target Analysis
explorer.target_analysis(TARGET_COLUMN)

# 3. Missing Value Analysis
missing_percentage = explorer.missing_value_analysis()

print("\nMissing Value Percentage:")
print(missing_percentage)

# 4. Numerical Analysis
numerical_summary = explorer.numerical_analysis()

print("\nNumerical Summary:")
print(numerical_summary)

# 5. Categorical Analysis
explorer.categorical_analysis()

# 6. Correlation Analysis
correlation_matrix = explorer.correlation_analysis()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================

preprocessor = DataPreprocessor(df)

# 1. Drop columns with more than 46% missing values
preprocessor.drop_high_missing_columns()

# 2. Create EXT_SOURCE_1 missing indicator
preprocessor.create_ext_source_missing_indicator()

# 3. Handle OCCUPATION_TYPE missing values
preprocessor.handle_occupation_type()

# 4. Handle NAME_TYPE_SUITE missing values
preprocessor.handle_name_type_suite()

# 5. Handle remaining numerical missing values
preprocessor.handle_numerical_missing_values()

# 6. Handle remaining categorical missing values
preprocessor.handle_categorical_missing_values()

# 7. Check DAYS_EMPLOYED special value
# Do NOT replace 365243 with NaN
preprocessor.check_days_employed_special_value()

# 8. Cap CNT_CHILDREN at 6
preprocessor.cap_children()

# 9. Apply IQR-based outlier capping
preprocessor.cap_outliers()


# ============================================================
# GET FINAL PREPROCESSED DATA
# ============================================================

df1 = preprocessor.get_data()


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("Original Dataset Shape:", df.shape)
print("Preprocessed Dataset Shape:", df1.shape)

print("\nTotal Missing Values:")
print(df1.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df1.duplicated().sum())

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================

df1 = preprocessor.get_data()
feature_engineer = FeatureEngineer(df1)

# 2.1 Age Features
feature_engineer.create_age_features()

# 2.2 Employment Features
feature_engineer.create_employment_features()

# 2.3 Income Features
feature_engineer.create_income_features()

# 2.4 to 2.7 Loan Features
feature_engineer.create_loan_features()

# 2.8 Family Size Feature
feature_engineer.create_family_size_feature()

# 2.9 Children Feature
feature_engineer.create_children_feature()

# 2.10 Residential Features
feature_engineer.create_residential_features()

# 2.11 Vehicle Ownership
feature_engineer.create_vehicle_ownership_feature()

# 2.12 Property Ownership
feature_engineer.create_property_ownership_feature()

# 2.13 Bureau Inquiry Features
feature_engineer.create_bureau_inquiry_features()

# 2.14 External Credit Score Features
feature_engineer.create_external_credit_features()

# 2.15 Property Quality Feature
feature_engineer.create_property_quality_feature()

# 2.16 Contact Verification Feature
feature_engineer.create_contact_verification_feature()

# 2.17 Document Completeness Feature
feature_engineer.create_document_completeness_feature()

# 2.18 Social Risk Feature
feature_engineer.create_social_risk_feature()

# Get final engineered dataset
df2 = feature_engineer.get_data()

# ============================================================
# FEATURE ENGINEERING VALIDATION
# ============================================================

new_features = [

    # Age Features
    "AGE_YEARS",
    "AGE_GROUP",

    # Employment Features
    "EMPLOYMENT_DATA_UNAVAILABLE",
    "EMPLOYMENT_YEARS",
    "EMPLOYMENT_STABILITY",

    # Income and Loan Features
    "INCOME_CATEGORY",
    "LTI_RATIO",
    "EMI_INCOME_RATIO",
    "LTV_RATIO",
    "DISPOSABLE_MONTHLY_INCOME",

    # Family Features
    "FAMILY_SIZE_CATEGORY",
    "CHILDREN_CATEGORY",

    # Residential Features
    "REGISTRATION_YEARS",
    "RESIDENTIAL_STABILITY",

    # Ownership Features
    "OWNS_VEHICLE",
    "OWNS_PROPERTY",

    # Bureau Inquiry Features
    "TOTAL_BUREAU_INQUIRIES",
    "RECENT_BUREAU_INQUIRIES",

    # External Credit Score Features
    "AVG_EXTERNAL_SCORE",
    "MAX_EXTERNAL_SCORE",
    "MIN_EXTERNAL_SCORE",

    # Property Feature
    "PROPERTY_QUALITY_SCORE",

    # Verification Features
    "CONTACT_VERIFICATION_SCORE",
    "DOCUMENT_COMPLETENESS_SCORE",

    # Social Risk Feature
    "SOCIAL_RISK_SCORE"
]

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("Input Dataset Shape:", df1.shape)
print("Final Dataset Shape:", df2.shape)

print("\nTotal Missing Values:")
print(df2.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df2.duplicated().sum())

print("\nMissing Engineered Features:")
print([
    col for col in new_features
    if col not in df2.columns
])

# ============================================================
# STEP 5: FEATURE ENCODING
# ============================================================

feature_encoder = FeatureEncoder(df2)

feature_encoder.encode_binary_features()

feature_encoder.encode_onehot_features()

feature_encoder.encode_gender()

feature_encoder.encode_ordinal_features()

feature_encoder.encode_employment_stability()

feature_encoder.encode_frequency_features()


# Get encoded dataset
df_encoded = feature_encoder.get_data()

# ============================================================
# FINAL ENCODING VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENCODING COMPLETED")
print("=" * 60)

print("Final Dataset Shape:", df_encoded.shape)

print("\nRemaining Object Columns:")
print(
    df_encoded.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
)

print("\nMissing Values:")
print(
    df_encoded.isnull().sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
)

print("\nDuplicate Rows:")
print(df_encoded.duplicated().sum())

# ============================================================
# STEP 9C: CREATE MASTER COPY FOR FEATURE SELECTION / MODELING
# ============================================================

df_model = df_encoded.copy()

print("\nEncoded Dataset Shape:", df_encoded.shape)
print("Model Dataset Shape:", df_model.shape)

# ============================================================
# STEP 9D: DEFINE FEATURES AND TARGET
# ============================================================

# Target variable
y = df_model["TARGET"].copy()

# Feature variables
X = df_model.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR"
    ]
)

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# ============================================================
# STEP 10: FEATURE SCALING
# ============================================================

feature_scaler = FeatureScaler(X)

# View numerical columns
feature_scaler.get_numerical_columns()

# View columns selected for scaling
feature_scaler.show_scale_columns()

# Perform scaling
feature_scaler.scale_features()

# Get scaled feature matrix
X_scaled = feature_scaler.get_data()

# ============================================================
# FEATURE SCALING VALIDATION
# ============================================================

scaling_check = feature_scaler.validate_scaling()

print("\n" + "=" * 60)
print("FEATURE SCALING VALIDATION")
print("=" * 60)

print(scaling_check)

# ============================================================
# STEP 11: FEATURE SELECTION
# ============================================================

feature_selector = FeatureSelector(X, y)

# ------------------------------------------------------------
# STEP 11.1: Correlation Analysis
# ------------------------------------------------------------

feature_selector.correlation_analysis(
    threshold=0.90
)

feature_selector.remove_redundant_features()

# ------------------------------------------------------------
# STEP 11.2: Information Value / WOE
# ------------------------------------------------------------

feature_selector.prepare_iv_data()

feature_selector.calculate_iv()

feature_selector.select_iv_features(
    threshold=0.02
)

# ------------------------------------------------------------
# STEP 11.3: Chi-Square Test
# ------------------------------------------------------------

feature_selector.chi_square_test(
    max_unique_values=10,
    significance_level=0.05
)

# ------------------------------------------------------------
# STEP 11.4: Mutual Information
# ------------------------------------------------------------

feature_selector.mutual_information(
    top_n=40
)

# ------------------------------------------------------------
# STEP 11.5: Recursive Feature Elimination
# ------------------------------------------------------------

feature_selector.recursive_feature_elimination(
    sample_size=50000,
    n_features_to_select=40,
    step=5
)

# ------------------------------------------------------------
# STEP 11.6: Random Forest Feature Importance
# ------------------------------------------------------------

feature_selector.random_forest_importance(
    top_n=40
)

# ------------------------------------------------------------
# STEP 11.7: Create Final Features
# ------------------------------------------------------------

feature_selector.create_final_features()

# Get final dataset
X_final, y_final = (
    feature_selector.get_final_data()
)

# ============================================================
# FINAL FEATURE SELECTION VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL FEATURE SELECTION COMPLETED")
print("=" * 60)

print("\nFinal Feature Matrix Shape:")
print(X_final.shape)

print("\nFinal Target Shape:")
print(y_final.shape)

print("\nTotal Missing Values:")
print(X_final.isnull().sum().sum())

print("\nDuplicate Rows:")
print(X_final.duplicated().sum())

print("\nFinal Selected Features:")

for i, feature in enumerate(
    X_final.columns,
    start=1
):
    print(f"{i}. {feature}")

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.20,
    stratify=y_final,
    random_state=42
)

print("=" * 60)
print("TRAIN-TEST SPLIT COMPLETED")
print("=" * 60)

print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)

print("y_train Shape:", y_train.shape)
print("y_test Shape :", y_test.shape)


# ============================================================
# MODEL TRAINING
# ============================================================

model_trainer = ModelTrainer(
    X_train,
    X_test,
    y_train,
    y_test
)


# ------------------------------------------------------------
# 1. Logistic Regression
# ------------------------------------------------------------

model_trainer.train_logistic_regression()


# ------------------------------------------------------------
# 2. Decision Tree
# ------------------------------------------------------------

model_trainer.train_decision_tree()


# ------------------------------------------------------------
# 3. Random Forest
# ------------------------------------------------------------

model_trainer.train_random_forest()


# ============================================================
# CROSS VALIDATION - LOGISTIC REGRESSION
# ============================================================

logistic_cv = ModelCrossValidator(
    model=model_trainer.models["Logistic Regression"],
    model_name="Logistic Regression",
    X=X_train,
    y=y_train,
    n_splits=5
)

logistic_cv.run_cross_validation()

logistic_cv_summary = logistic_cv.get_summary()

print("\nLogistic Regression CV Summary:")
print(logistic_cv_summary)


# ============================================================
# CROSS VALIDATION - DECISION TREE
# ============================================================

decision_tree_cv = ModelCrossValidator(
    model=model_trainer.models["Decision Tree"],
    model_name="Decision Tree",
    X=X_train,
    y=y_train,
    n_splits=5
)

decision_tree_cv.run_cross_validation()

decision_tree_cv_summary = decision_tree_cv.get_summary()

print("\nDecision Tree CV Summary:")
print(decision_tree_cv_summary)


# ============================================================
# CROSS VALIDATION - RANDOM FOREST
# ============================================================

rf_cv = ModelCrossValidator(
    model=model_trainer.models["Random Forest"],
    model_name="Random Forest",
    X=X_train,
    y=y_train,
    n_splits=5
)

rf_cv.run_cross_validation()

rf_cv_summary = rf_cv.get_summary()

print("\nRandom Forest CV Summary:")
print(rf_cv_summary)


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

model_trainer.hyperparameter_tuning()


# ============================================================
# MODEL EVALUATION
# ============================================================

# ------------------------------------------------------------
# 1. LOGISTIC REGRESSION EVALUATION
# ------------------------------------------------------------

logistic_evaluator = ModelEvaluator(
    model=model_trainer.models["Logistic Regression"],
    model_name="Logistic Regression",
    X_test=X_test,
    y_test=y_test
)

logistic_evaluator.evaluate()

logistic_evaluator.calculate_ks_gini()

logistic_evaluator.plot_confusion_matrix()

logistic_evaluator.plot_roc_curve()

logistic_evaluator.plot_precision_recall_curve()

# ------------------------------------------------------------
# 2. DECISION TREE EVALUATION
# ------------------------------------------------------------

decision_tree_evaluator = ModelEvaluator(
    model=model_trainer.models["Decision Tree"],
    model_name="Decision Tree",
    X_test=X_test,
    y_test=y_test
)

decision_tree_evaluator.evaluate()

decision_tree_evaluator.calculate_ks_gini()

decision_tree_evaluator.plot_confusion_matrix()

decision_tree_evaluator.plot_roc_curve()

decision_tree_evaluator.plot_precision_recall_curve()

# ------------------------------------------------------------
# 3. RANDOM FOREST EVALUATION
# ------------------------------------------------------------

rf_evaluator = ModelEvaluator(
    model=model_trainer.models["Random Forest"],
    model_name="Random Forest",
    X_test=X_test,
    y_test=y_test
)

rf_evaluator.evaluate()

rf_evaluator.calculate_ks_gini()

rf_evaluator.plot_confusion_matrix()

rf_evaluator.plot_roc_curve()

rf_evaluator.plot_precision_recall_curve()

# ------------------------------------------------------------
# 4. TUNED RANDOM FOREST EVALUATION
# ------------------------------------------------------------

tuned_rf_evaluator = ModelEvaluator(
    model=model_trainer.models["Tuned Random Forest"],
    model_name="Tuned Random Forest",
    X_test=X_test,
    y_test=y_test
)

tuned_rf_evaluator.evaluate()

tuned_rf_evaluator.calculate_ks_gini()

tuned_rf_evaluator.plot_confusion_matrix()

tuned_rf_evaluator.plot_roc_curve()

tuned_rf_evaluator.plot_precision_recall_curve()

# ============================================================
# SHAP ANALYSIS
# ============================================================

shap_analyzer = SHAPAnalyzer(
    model=model_trainer.best_model,
    X_train=model_trainer.X_train,
    X_test=model_trainer.X_test,
    sample_size=5000,
    random_state=42
)

print("SHAP Analyzer Created Successfully!")

# ============================================================
# STEP 1: CREATE SHAP SAMPLE
# ============================================================

shap_analyzer.create_sample()

# ============================================================
# STEP 2: CREATE SHAP EXPLAINER
# ============================================================

shap_analyzer.create_explainer()

# ============================================================
# STEP 3: CALCULATE SHAP VALUES
# ============================================================

shap_analyzer.calculate_shap_values()

# ============================================================
# STEP 4: SHAP SUMMARY PLOT
# ============================================================

shap_analyzer.summary_plot()

# ============================================================
# STEP 5: SHAP FEATURE IMPORTANCE
# ============================================================

shap_importance = (
    shap_analyzer.feature_importance()
)

# ============================================================
# STEP 6: SHAP FEATURE IMPORTANCE BAR PLOT
# ============================================================

shap_analyzer.feature_importance_plot()

# ============================================================
# PREDICTION MODEL
# ============================================================

from prediction import Prediction


prediction_model = Prediction(
    model=model_trainer.best_model,
    feature_names=X_train.columns.tolist()
)

print("\n" + "=" * 60)
print("PREDICTION MODEL CREATED")
print("=" * 60)

print(
    "Number of Expected Features:",
    len(X_train.columns)
)

print("\nFinal Model Features:")

for i, feature in enumerate(
    X_train.columns,
    start=1
):
    print(f"{i:2}. {feature}")