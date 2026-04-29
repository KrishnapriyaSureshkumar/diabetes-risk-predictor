import streamlit as st
import pandas as pd
import pickle

# PAGE CONFIG
st.set_page_config(page_title="Healthcare Decision Support System", layout="centered")

# LOAD MODEL
saved = pickle.load(open("diabetes_pipeline.pkl", "rb"))
model = saved["model"]
feature_names = saved["features"]
importance = saved["importance"]
model_name = saved["model_name"]
data = saved["data"]

# TITLE
st.title("Diabetes Risk Prediction System")
st.markdown(f"### Model Used: {model_name}")

st.warning("⚠️ This system provides predictive insights and is intended for decision support only.")

# ======================
# DATA INSIGHTS
# ======================
st.subheader("Dataset Insights")
st.bar_chart(data["Outcome"].value_counts())

# ======================
# FEATURE IMPORTANCE VISUAL
# ======================
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": list(importance.keys()),
    "Importance": list(importance.values())
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.write(f"Key contributing factors: {importance_df.head(3)['Feature'].tolist()}")

# ======================
# EXTRA INSIGHT
# ======================
st.subheader("Clinical Insight")

st.write("""
- Glucose and BMI are typically strong indicators of diabetes risk  
- Age and Insulin levels also influence prediction  
- Model uses machine learning to identify hidden patterns in patient data  
""")

# ======================
# INPUT SECTION
# ======================
st.subheader("Enter Patient Clinical Details")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 25)

# ======================
# PREDICTION
# ======================
if st.button("Predict Risk"):

    if glucose == 0 or bp == 0 or bmi == 0:
        st.error("Invalid clinical values detected!")
    else:
        input_data = pd.DataFrame([{
            'Pregnancies': preg,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }])

        proba = model.predict_proba(input_data)
        risk = proba[0][1] * 100

        st.write(f"### Predicted Diabetes Risk: {risk:.2f}%")
        st.progress(int(risk))

        if risk < 30:
            st.success("Low Risk")
        elif risk < 60:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")
