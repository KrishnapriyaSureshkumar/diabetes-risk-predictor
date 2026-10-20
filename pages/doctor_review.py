
import streamlit as st

from prediction_service import predict_compact_model

st.set_page_config(
    page_title="Doctor Review",
    page_icon="🩺",
    layout="wide",
)

st.title("Doctor Review")
st.caption("Structured patient intake summary")

if (
    "intake" not in st.session_state
    or not st.session_state.intake.get("completed")
):
    st.info("No completed intake is available. Complete Patient Intake first.")

    if st.button("Go to Patient Intake"):
        st.switch_page("app.py")

    st.stop()

d = st.session_state.intake

st.subheader("Intake status")
st.write("**Status:** Ready for review")
st.write("**Created:**", d.get("created_at", "Not recorded"))

st.subheader("Patient-reported information")

history_fields = [
    ("Main concern", "chief_concern"),
    ("Symptom timeline", "symptom_timeline"),
    ("Additional symptoms", "symptom_details"),
    ("Medical history", "medical_history"),
    ("Medications / supplements", "medications"),
    ("Allergies", "allergies"),
    ("Previous tests", "previous_tests"),
]

for label, key in history_fields:
    st.markdown(f"**{label}:** {d.get(key) or 'Not provided'}")

st.subheader("Measurements")

measurement_fields = [
    ("Gender", "gender"),
    ("Age (years)", "age"),
    ("Height (cm)", "height_cm"),
    ("Weight (kg)", "weight_kg"),
    ("Calculated BMI", "bmi"),
    ("BMI source", "bmi_source"),
    ("Glucose (mg/dL)", "glucose"),
    ("Pregnancy history", "pregnancy_status"),
    ("Number of pregnancies", "pregnancies"),
    ("Blood pressure", "blood_pressure"),
    ("Measurement notes", "measurement_notes"),
]

for label, key in measurement_fields:
    value = d.get(key)

    if value is None or value == "":
        value = "Not provided"

    st.write(f"**{label}:** {value}")

st.subheader("Model output")

result = predict_compact_model(d)

if result["status"] == "model_output_available":
    st.metric(
        "Predicted probability for Outcome = 1",
        f'{result["model_output"]:.3f}',
    )
    st.write("**Model:**", result["model"])
    st.write("**Features used:**", ", ".join(result["features"]))

elif result["status"] == "prediction_withheld":
    st.info(result["message"])

    if result.get("missing_fields"):
        st.write(
            "**Unavailable inputs:**",
            ", ".join(result["missing_fields"]),
        )

else:
    st.warning(result.get("message", "Model output unavailable."))

for note in result.get("notes", []):
    st.caption(note)

st.subheader("Review checklist")

checks = [
    "Verify measurements and units.",
    "Confirm symptoms, medications and allergies.",
    "Check previous test records when available.",
    "Review the intake information.",
]

for i, item in enumerate(checks):
    st.checkbox(item, key=f"review_check_{i}")

report_lines = [
    "PATIENT INTAKE SUMMARY",
    "",
    f"Created: {d.get('created_at', 'Not recorded')}",
    "",
    "PATIENT-REPORTED INFORMATION",
]

for label, key in history_fields:
    report_lines.append(f"{label}: {d.get(key) or 'Not provided'}")

report_lines.extend(["", "MEASUREMENTS"])

for label, key in measurement_fields:
    value = d.get(key)
    report_lines.append(
        f"{label}: {value if value is not None else 'Not provided'}"
    )

report_lines.extend([
    "",
    "MODEL OUTPUT",
    f"Status: {result.get('status', 'Unavailable')}",
    f"Model: {result.get('model', 'Not run')}",
    (
        "Predicted probability for Outcome = 1: "
        f"{result.get('model_output', 'Unavailable')}"
    ),
    (
        "Features used: "
        f"{', '.join(result.get('features', [])) or 'None'}"
    ),
    (
        "Unavailable inputs: "
        f"{', '.join(result.get('missing_fields', [])) or 'None'}"
    ),
])

report = "\n".join(report_lines)

st.download_button(
    "Download Patient Summary (.txt)",
    data=report,
    file_name="patient_intake_summary.txt",
    mime="text/plain",
)

st.divider()

if st.button("Back to Patient Intake"):
    st.switch_page("app.py")
