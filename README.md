# 🏠 Home Loan Application Scorecard

## 📌 Project Overview

The **Home Loan Application Scorecard** is a machine learning classification project that predicts the **probability of default** for home loan applicants.

The model analyzes applicant demographic, financial, employment, credit, property, and family-related information to assess credit risk.

The trained model is integrated with a **Streamlit web application** that allows users to enter applicant information and receive a risk prediction.

---

## 🎯 Business Objective

The objective of this project is to help financial institutions:

* Assess the credit risk of loan applicants.
* Predict the probability of loan default.
* Support faster and data-driven loan screening.
* Identify applicants with higher default risk.

---

## 📊 Dataset

This project uses the **Home Credit Default Risk** dataset.

The dataset contains information related to:

* Applicant demographics
* Income and loan details
* Employment history
* External credit scores
* Property information
* Family and social information

### Target Variable

`TARGET`

* `0` → No Default
* `1` → Default

This is a **binary classification problem**.

---

## ⚙️ Feature Engineering

Several meaningful features were created from the raw data.

### Employment Features

* `EMPLOYMENT_YEARS` — converts employment duration from days to years.
* `EMPLOYMENT_DATA_UNAVAILABLE` — identifies unavailable employment information.
* `EMPLOYMENT_STABILITY` — categorizes applicants into:

  * New Employee
  * Experienced Employee
  * Highly Experienced Employee
  * Data Unavailable

### Credit Features

External credit score variables were combined to create:

* `AVG_EXTERNAL_SCORE`
* `MAX_EXTERNAL_SCORE`
* `MIN_EXTERNAL_SCORE`

### Other Features

* `AGE_YEARS`
* `PROPERTY_QUALITY_SCORE`

These engineered features help the model capture meaningful applicant risk patterns.

---

## 🤖 Machine Learning Model

The final model used in the project is:

**Random Forest Classifier**

The model was selected to capture non-linear relationships and interactions between applicant features.

The trained model is saved using Joblib.

```text
models/
├── best_model.pkl
└── feature_names.pkl
```

The final model uses **40 selected features**.

---

## 📈 Model Evaluation

The model was evaluated using unseen data and cross-validation.

Key evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

> Added actual metric values here after checking your final model evaluation results.

Example:

| Metric    | Score    |
| --------- | ----:    |
| Accuracy  |   90.88% |
| Precision |   34.76% |
| Recall    |   14.84% |
| F1 Score  |   20.8%  |
| ROC-AUC   |   0.7413 |

---

## 🖥️ Streamlit Application

A Streamlit-based frontend was developed to provide an interactive prediction interface.

The application includes:

* Applicant Information
* Loan Information
* Credit Information
* Employment & Region
* Property Information
* Family & Social Information

The application accepts applicant details and returns the predicted **default probability and risk assessment**.

### 📸 Application Screenshot

Add your Streamlit screenshot here:

```text
Screen_shots/
└── home_loan_scorecard.png
```

Then use:

```markdown
![Home Loan Scorecard](Screen_shots/home_loan_scorecard.png)
```

---

## 📁 Project Structure

```text
Home_Loan_Scorecard/
│
├── app.py
├── main.py
├── config.py
│
├── data_loader.py
├── data_explorer.py
├── data_preprocessing.py
│
├── feature_engineering.py
├── feature_encoding.py
├── feature_scaling.py
├── feature_selection.py
│
├── model_training.py
├── model_cross_validation.py
├── model_evaluator.py
│
├── shap_analyzer.py
├── inference_preprocessor.py
├── prediction.py
│
├── check_model.py
├── requirements.txt
│
├── models/
│   ├── best_model.pkl
│   └── feature_names.pkl
│
└── notebooks/
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **SHAP**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**
* **VS Code**
* **Git & GitHub**

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Snehal18gitHub/Home_Loan_Scorecard_Development_MachineLearningModel.git
```

### 2. Navigate to the project

```bash
cd Home_Loan_Scorecard_Development_MachineLearningModel
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 🔍 Model Explainability

**SHAP (SHapley Additive exPlanations)** was used to understand feature contributions and improve model interpretability.

This helps identify which applicant characteristics have the greatest influence on the model's prediction.

---

## 🚀 Key Highlights

* End-to-end machine learning project.
* Real-world credit risk use case.
* Extensive feature engineering.
* 40 selected model features.
* Random Forest classification.
* Cross-validation and model evaluation.
* SHAP-based model explainability.
* Modular Python project structure.
* Interactive Streamlit prediction application.
* Model serialization using Joblib.

---

## 👩‍💻 Author

**Snehal Kapkar**

Data Science | Machine Learning | Python
