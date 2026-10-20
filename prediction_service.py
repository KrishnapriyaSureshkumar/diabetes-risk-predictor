
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "missing_diabetes_models.pkl"

COMPACT_FEATURES = [
    "Age",
    "Glucose",
    "BMI",
    "Pregnancies",
]


def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI using height in cm and weight in kg."""
    try:
        height_cm = float(height_cm)
        weight_kg = float(weight_kg)
    except (TypeError, ValueError):
        return None

    if not (0 < height_cm <= 250 and 0 < weight_kg <= 400):
        return None

    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)


def predict_compact_model(intake):
    """Run the saved compact model when required inputs are available."""

    values = {
        "Age": intake.get("age"),
        "Glucose": intake.get("glucose"),
        "BMI": intake.get("bmi"),
        "Pregnancies": intake.get("pregnancies"),
    }

    missing = []
    notes = []

    for feature, value in values.items():
        if value is None or value == "":
            missing.append(feature)
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            notes.append(f"{feature} must be numeric.")
            continue

        if not pd.notna(number) or number < 0:
            notes.append(
                f"{feature} must be a valid non-negative number."
            )
            continue

        if feature == "Age" and not 1 <= number <= 120:
            notes.append("Age must be between 1 and 120.")

        elif feature == "Glucose" and not 1 <= number <= 1000:
            notes.append(
                "Glucose must be a positive value in mg/dL."
            )

        elif feature == "BMI" and not 5 <= number <= 100:
            notes.append(
                "BMI is outside the expected input range. "
                "Verify height and weight."
            )

        elif feature == "Pregnancies":
            if number > 30 or not number.is_integer():
                notes.append(
                    "Pregnancies must be a whole number from 0 to 30."
                )

    if notes:
        return {
            "status": "needs_correction",
            "message": "Review the input values.",
            "missing_fields": missing,
            "notes": notes,
        }

    if missing:
        return {
            "status": "prediction_withheld",
            "message": (
                "The model requires Age, Glucose, BMI and "
                "Pregnancies. One or more inputs are unavailable."
            ),
            "missing_fields": missing,
            "notes": [],
        }

    if not MODEL_PATH.exists():
        return {
            "status": "prediction_withheld",
            "message": f"Model file not found: {MODEL_PATH}",
            "missing_fields": [],
            "notes": [],
        }

    try:
        saved_models = joblib.load(MODEL_PATH)

        bundle = saved_models[
            ("zeros_as_missing", "compact_4")
        ]

        model = bundle["model"]
        features = bundle.get("features", COMPACT_FEATURES)

        row = pd.DataFrame(
            [[float(values[feature]) for feature in features]],
            columns=features,
        )

        probability = float(model.predict_proba(row)[0, 1])

    except Exception as exc:
        return {
            "status": "prediction_withheld",
            "message": (
                "The saved model could not be loaded or evaluated."
            ),
            "missing_fields": [],
            "notes": [
                f"Technical detail: {type(exc).__name__}: {exc}"
            ],
        }

    return {
        "status": "model_output_available",
        "model": "compact_4 (zeros_as_missing)",
        "model_output": probability,
        "features": list(features),
        "missing_fields": [],
        "notes": [],
    }
