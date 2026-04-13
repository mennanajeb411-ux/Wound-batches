import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

st.set_page_config(page_title="Smart Wound Healing AI", layout="centered")
st.title("🩹 Smart Wound Healing AI Platform")
st.write("أدخل بيانات الجرح وسيوصي النموذج بالعلاج المناسب تلقائيًا.")

# بيانات التدريب
data = {
    "Temperature": [38.5, 36.7, 37.8, 39.1, 35.9, 37.0, 38.2, 36.4, 39.0, 35.5],
    "pH": [6.2, 7.4, 5.9, 6.1, 7.2, 6.5, 6.0, 7.3, 5.7, 7.1],
    "Moisture": [85, 45, 92, 88, 50, 70, 80, 55, 95, 40],
    "Infection": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    "Diabetic": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "Wound Type": ["Acute", "Superficial", "Chronic", "Acute", "Post-surgical", 
                   "Superficial", "Chronic", "Post-surgical", "Acute", "Superficial"],
    "Healing Fast": ["Slow", "Fast", "Slow", "Slow", "Fast", 
                     "Fast", "Slow", "Fast", "Slow", "Fast"],
    "Gender": ["Male","Female","Male","Female","Male","Female","Male","Female","Male","Female"],
    "Age": [65, 28, 70, 35, 55, 24, 67, 40, 73, 30],
    "Recommended Treatment": [
        "Patch with Stem Cells+ Antibiotic", "Patch with Stem Cells only",
        "Patch with Strong Antimicrobial+ GF", "Patch with Collagen+ Antibiotic",
        "Patch with Collagen only", "Patch with Stem Cells only",
        "Patch with Strong Antimicrobial+ GF", "Patch with Collagen only",
        "Patch with Stem Cells+ Antibiotic", "Patch with Collagen+ Antibiotic"
    ]
}
df = pd.DataFrame(data)

# ترميز البيانات
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Recommended Treatment", axis=1)
y = df["Recommended Treatment"]

# تدريب النموذج
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# واجهة المستخدم
st.subheader("📋 أدخل بيانات المريض")

col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("درجة حرارة الجرح (°C)", min_value=35.0, max_value=40.0, value=37.5, step=0.1)
    ph = st.number_input("مستوى الحموضة (pH)", min_value=5.0, max_value=8.0, value=6.5, step=0.1)
    moisture = st.number_input("نسبة الرطوبة (%)", min_value=20, max_value=100, value=70)
    infection = st.selectbox("هل يوجد عدوى؟", ["No", "Yes"])
    diabetic = st.selectbox("هل المريض مصاب بالسكري؟", ["No", "Yes"])

with col2:
    wound_type = st.selectbox("نوع الجرح", ["Acute", "Superficial", "Chronic", "Post-surgical"])
    healing_fast = st.selectbox("سرعة الالتئام", ["Slow", "Fast"])
    gender = st.selectbox("الجنس", ["Male", "Female"])
    age = st.number_input("العمر", min_value=1, max_value=100, value=40)

if st.button("🔍 احصل على التوصية"):
    # تحضير البيانات للتنبؤ
    input_data = {
        "Temperature": [temp],
        "pH": [ph],
        "Moisture": [moisture],
        "Infection": [1 if infection == "Yes" else 0],
        "Diabetic": [1 if diabetic == "Yes" else 0],
        "Wound Type": [label_encoders["Wound Type"].transform([wound_type])[0]],
        "Healing Fast": [label_encoders["Healing Fast"].transform([healing_fast])[0]],
        "Gender": [label_encoders["Gender"].transform([gender])[0]],
        "Age": [age]
    }
    
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)
    treatment = label_encoders["Recommended Treatment"].inverse_transform(prediction)[0]
    
    st.success(f"✅ العلاج الموصى به: **{treatment}**")
    st.info("⚠️ هذه التوصية من نموذج الذكاء الاصطناعي ولا تغني عن استشارة الطبيب")