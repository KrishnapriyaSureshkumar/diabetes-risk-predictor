# AI-Based Diabetes Risk Prediction and Clinical Reporting System

## Overview

This project is a machine learning-assisted application that combines conversational patient information intake, clinical parameter collection, and diabetes classification output in a Streamlit web interface.

The application collects patient-reported information, records available clinical measurements, calculates BMI from height and weight, and provides a clinician-facing review page with a model-generated probability score.

The project demonstrates how machine learning can be integrated into a structured patient intake and clinical reporting workflow.

## Features

- **Conversational Patient Intake:** Collects the patient's main concern, symptom timeline, medical history, medications, allergies, and previous test information.
- **Clinical Data Collection:** Records age, height, weight, glucose level, gender, pregnancy history where applicable, and optional blood pressure information.
- **Automatic BMI Calculation:** Calculates BMI from the entered height and weight.
- **Machine Learning Prediction:** Uses a saved Logistic Regression model to generate a diabetes classification probability.
- **Clinician Review:** Displays collected patient information, available measurements, and model output on a separate review page.
- **Input Validation:** Checks required model inputs and identifies missing or invalid values before prediction.
- **Report Export:** Allows the clinician-facing summary to be downloaded as a text file.
- **Interactive Web Interface:** Built with Streamlit for a straightforward, multi-step intake workflow.

## Tech Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib

## Project Structure

```text
diabetes-risk-predictor/
├── app.py
├── prediction_service.py
├── requirements.txt
├── models/
│   └── missing_diabetes_models.pkl
└── pages/
    └── doctor_review.py
```

- `app.py` — Implements the conversational intake workflow and clinical measurement form.
- `prediction_service.py` — Validates model inputs, calculates BMI, and obtains the saved model's output.
- `models/missing_diabetes_models.pkl` — Contains the serialized model bundle used for prediction.
- `pages/doctor_review.py` — Displays the collected information, model output, and downloadable review summary.
- `requirements.txt` — Lists the Python dependencies required to run the application.

## Dataset and Model

The project builds on the Healthcare Diabetes dataset available on Kaggle.

The current compact prediction model uses four input features:

- Age
- Glucose
- BMI
- Pregnancies

The application loads the saved Logistic Regression model rather than training a new model every time it starts.

The model's output is a predicted class probability. It should not be interpreted as a calibrated estimate of an individual's personal diabetes risk.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/KrishnapriyaSureshkumar/diabetes-risk-predictor.git
cd diabetes-risk-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## Application Workflow

1. The user answers conversational questions about their health concerns and relevant medical history.
2. The user selects a gender option and enters available clinical measurements.
3. The application calculates BMI when valid height and weight values are available.
4. After intake completion, the user opens the clinician review page.
5. The application validates the inputs required by the compact model.
6. If the required inputs are available and valid, the saved model generates a classification probability.
7. The clinician-facing summary can be downloaded for review.

## Limitations

- The application is an academic prototype and is not a clinically validated diagnostic tool.
- The model uses only four features and does not represent the full range of factors associated with diabetes.
- The predicted class probability is model-specific and should not be treated as an individual's confirmed diabetes risk.
- Missing or unsuitable inputs can prevent the application from generating a model output.
- Model performance and generalizability depend on the dataset, preprocessing, training procedure, and evaluation methodology.
- The application does not replace professional medical assessment or laboratory testing.

## Future Improvements

- Evaluate model performance using appropriate classification metrics and independent test data.
- Compare Logistic Regression with additional classification algorithms.
- Improve probability calibration and evaluate performance across relevant patient groups.
- Expand input validation and improve handling of unavailable or inapplicable measurements.
- Enhance the clinician review interface and report formatting.
- Add automated testing, model versioning, and reproducible training procedures.
- Explore secure authentication and privacy-conscious handling of patient information.

## Disclaimer

This project is intended for educational and demonstration purposes only. Its predictions must not be used as the sole basis for diagnosis, treatment, or other clinical decisions.
