
import streamlit as st
from datetime import datetime

from prediction_service import calculate_bmi

st.set_page_config(
    page_title="Patient Intake Assistant",
    page_icon="💬",
    layout="centered",
)

st.title("Patient Intake Assistant")
st.caption("Conversational patient information intake")


def new_intake():
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "chief_concern": "",
        "symptom_timeline": "",
        "symptom_details": "",
        "medical_history": "",
        "medications": "",
        "allergies": "",
        "previous_tests": "",
        "gender": None,
        "age": None,
        "height_cm": None,
        "weight_kg": None,
        "bmi": None,
        "bmi_source": "",
        "glucose": None,
        "pregnancies": None,
        "pregnancy_status": "Not answered",
        "blood_pressure": "",
        "measurement_notes": "",
        "completed": False,
    }


if "intake" not in st.session_state:
    st.session_state.intake = new_intake()

if "chat_step" not in st.session_state:
    st.session_state.chat_step = 0

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "measurement_step" not in st.session_state:
    st.session_state.measurement_step = 1


questions = [
    (
        "chief_concern",
        "Hi! What brings you in today? Tell me about your main concern in your own words.",
        "Describe your symptoms or anything you'd like the doctor to know.",
    ),
    (
        "symptom_timeline",
        "When did this start? Has it been getting worse, staying about the same, or coming and going?",
        "If you're unsure, it's okay to say so.",
    ),
    (
        "symptom_details",
        "Is there anything else about your symptoms that you think the doctor should know?",
        "You can type 'skip' if there's nothing else to add.",
    ),
    (
        "medical_history",
        "Do you have any existing medical conditions or relevant medical history?",
        "For example, high blood pressure or a previous diabetes diagnosis.",
    ),
    (
        "medications",
        "Do you currently take any medicines or supplements?",
        "Share their names if you know them. You can say 'none' or 'not sure'.",
    ),
    (
        "allergies",
        "Do you have any known allergies, including medicine allergies?",
        "You can say 'none known' or 'not sure'.",
    ),
    (
        "previous_tests",
        "Have you had any relevant tests recently?",
        "Share any results you remember. It's fine if you don't remember the details.",
    ),
]


data = st.session_state.intake

# Show the conversation history.
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# PATIENT INTAKE
# --------------------------------------------------

if not data["completed"]:

    step = st.session_state.chat_step

    # Stage 1: Conversational questions.
    if step < len(questions):
        key, prompt, hint = questions[step]

        with st.chat_message("assistant"):
            st.markdown(prompt)
            st.caption(hint)

        answer = st.chat_input("Type your answer...")

        if answer is not None:
            answer = answer.strip()

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": prompt,
            })

            st.session_state.chat_messages.append({
                "role": "user",
                "content": answer or "Not provided",
            })

            data[key] = (
                "Not provided"
                if answer.lower() == "skip" or not answer
                else answer
            )

            st.session_state.intake = data
            st.session_state.chat_step += 1
            st.rerun()

    # Stage 2: Ask gender separately.
    else:
        st.subheader("Patient details and measurements")

        if st.session_state.measurement_step == 1:
            gender_options = [
                "Female",
                "Male",
                "Other",
                "Prefer not to say",
            ]

            saved_gender = data.get("gender")

            gender_index = (
                gender_options.index(saved_gender)
                if saved_gender in gender_options
                else 0
            )

            gender = st.selectbox(
                "Gender",
                gender_options,
                index=gender_index,
                key="intake_gender",
            )

            if st.button(
                "Continue to measurements",
                type="primary",
            ):
                data["gender"] = gender
                st.session_state.intake = data
                st.session_state.measurement_step = 2
                st.rerun()

        # Stage 3: All measurements in a single form.
        else:
            gender = data.get("gender") or "Prefer not to say"

            with st.form("measurement_form"):
                c1, c2 = st.columns(2)

                age = c1.number_input(
                    "Age (years)",
                    min_value=0,
                    max_value=120,
                    value=int(data.get("age") or 0),
                    step=1,
                    help="Enter 0 if unknown.",
                )

                height = c2.number_input(
                    "Height (cm)",
                    min_value=0.0,
                    max_value=250.0,
                    value=float(data.get("height_cm") or 0),
                    step=0.5,
                    help="Enter 0 if unknown.",
                )

                c3, c4 = st.columns(2)

                weight = c3.number_input(
                    "Weight (kg)",
                    min_value=0.0,
                    max_value=400.0,
                    value=float(data.get("weight_kg") or 0),
                    step=0.5,
                    help="Enter 0 if unknown.",
                )

                glucose = c4.number_input(
                    "Glucose (mg/dL)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(data.get("glucose") or 0),
                    step=1.0,
                    help="Enter a known result in mg/dL.",
                )

                # Pregnancy question is skipped for Male.
                if gender == "Male":
                    pregnancy_status = "Not applicable (male)"
                    pregnancies = 0

                else:
                    pregnancy_status = st.radio(
                        "Pregnancy history",
                        [
                            "No previous pregnancies",
                            "Previous pregnancies",
                            "Not applicable / unsure",
                            "Prefer not to answer",
                        ],
                    )

                    pregnancies = None

                    if pregnancy_status == "No previous pregnancies":
                        pregnancies = 0

                    elif pregnancy_status == "Previous pregnancies":
                        pregnancies = st.number_input(
                            "Number of previous pregnancies",
                            min_value=1,
                            max_value=30,
                            value=1,
                            step=1,
                        )

                blood_pressure = st.text_input(
                    "Blood pressure (if known)",
                    value=(
                        data.get("blood_pressure", "")
                        if data.get("blood_pressure") != "Not provided"
                        else ""
                    ),
                    placeholder="e.g. 120/80",
                )

                measurement_notes = st.text_area(
                    "Optional measurement notes",
                    value=(
                        data.get("measurement_notes", "")
                        if data.get("measurement_notes") != "Not provided"
                        else ""
                    ),
                    placeholder=(
                        "e.g. Glucose was measured at a clinic last week."
                    ),
                )

                submitted = st.form_submit_button(
                    "Finish intake",
                    type="primary",
                )

            if submitted:
                data["gender"] = gender
                data["age"] = int(age) if age > 0 else None
                data["height_cm"] = height if height > 0 else None
                data["weight_kg"] = weight if weight > 0 else None
                data["glucose"] = glucose if glucose > 0 else None

                data["pregnancy_status"] = pregnancy_status
                data["pregnancies"] = (
                    int(pregnancies)
                    if pregnancies is not None
                    else None
                )

                data["blood_pressure"] = (
                    blood_pressure.strip() or "Not provided"
                )

                data["measurement_notes"] = (
                    measurement_notes.strip() or "Not provided"
                )

                # Calculate BMI automatically from height and weight.
                data["bmi"] = calculate_bmi(
                    data["height_cm"],
                    data["weight_kg"],
                )

                data["bmi_source"] = (
                    "Calculated from entered height and weight"
                    if data["bmi"] is not None
                    else "Unavailable: height and/or weight not provided"
                )

                data["completed"] = True
                st.session_state.intake = data
                st.rerun()


# --------------------------------------------------
# COMPLETED INTAKE
# --------------------------------------------------

else:
    st.success("Intake completed.")
    st.write("Your information is ready for review.")

    if st.button(
        "Open Doctor Review",
        type="primary",
    ):
        st.switch_page("pages/doctor_review.py")

    if st.button("Start a new intake"):
        st.session_state.intake = new_intake()
        st.session_state.chat_step = 0
        st.session_state.chat_messages = []
        st.session_state.measurement_step = 1

        # Clear widget state from the previous intake.
        for key in [
            "intake_gender",
            "intake_pregnancy_status",
            "intake_pregnancy_count",
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()
