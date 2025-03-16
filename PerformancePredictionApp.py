import streamlit as st
import joblib
import numpy as np

# Load models and encoders
model = joblib.load(r'C:\Users\HP\PerformanceModel.pkl')
le_job = joblib.load(r'C:\Users\HP\LabelEncoder_Job.pkl')
le_el = joblib.load(r'C:\Users\HP\le_Education_Level.pkl')
scaler = joblib.load(r'C:\Users\HP\StandardScaler1.pkl')

st.title("Employee Performance Prediction")
st.write("Fill in the employee details below to predict their performance score.")

# Layout for better UI
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("🏢 Job Title", ['Analyst', 'Consultant', 'Developer', 'Engineer', 'Manager', 'Specialist', 'Technician'])
    years_at_company = st.number_input("📅 Years at Company", min_value=0, max_value=50, value=5)
    education_level = st.selectbox("🎓 Education Level", ["Bachelor", "High School", "Master", "PhD"])
    monthly_salary = st.number_input("💰 Monthly Salary ($)", min_value=1000, max_value=20000, value=5000, step=500)

with col2:
    projects_handled = st.number_input("📊 Projects Handled", min_value=0, max_value=50, value=5)
    overtime_hours = st.number_input("⏰ Overtime Hours", min_value=0, max_value=50, value=5)
    remote_work_frequency = st.selectbox("🏠 Remote Work Frequency (%)", [0, 25, 50, 75, 100])
    promotions = st.number_input("🚀 Promotions", min_value=0, max_value=10, value=1)

# Work hours and training hours on the same row
col3, col4 = st.columns(2)
with col3:
    training_hours = st.slider("📚 Training Hours", min_value=0, max_value=150, value=10)
with col4:
    work_hours_per_week = st.slider("⏳ Work Hours/Week", min_value=20, max_value=80, value=40)

# Employee satisfaction with full-width slider
employee_satisfaction_score = st.slider("😊 Employee Satisfaction Score", min_value=1, max_value=4, value=2)

# Encode categorical inputs
job_encoded = le_job.transform([job_title])[0]
education_encoded = le_el.transform([education_level])[0]

# Prepare features
input_features = np.array([[job_encoded, years_at_company, education_encoded, monthly_salary,
                            work_hours_per_week, projects_handled, overtime_hours,
                            remote_work_frequency, training_hours, promotions,
                            employee_satisfaction_score]])

# Scale input
input_features_scaled = scaler.transform(input_features)

# Prediction
if st.button("🔮 Predict Performance"):
    prediction = model.predict(input_features_scaled)
    st.success(f"🎯 Predicted Performance Score of Employee: {prediction[0]}")

st.info("Tip: Input the employee's details thoroughly and click 'Predict Performance' to predict the employee performance score.")
