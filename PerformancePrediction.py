import streamlit as st
import joblib
import numpy as np

model=joblib.load(r'C:\Users\HP\PerformanceModel.pkl')
le_job=joblib.load(r'C:\Users\HP\LabelEncoder_Job.pkl')
le_edu=joblib.load(r'C:\Users\HP\LabelEncoder_Edu.pkl')
le_per=joblib.load(r'C:\Users\HP\LabelEncoder_Per.pkl')
le_rwf=joblib.load(r'C:\Users\HP\LabelEncoder_Rwf.pkl')
le_ess=joblib.load(r'C:\Users\HP\LabelEncoder_Ess.pkl')
scaler=joblib.load(r'C:\Users\HP\StandardScaler1.pkl')

st.title("Employee Performance Prediction")
st.write("Enter the employee details to predict their performance score")

job_title = st.selectbox("Job Ttile", ['Analyst','Consultant','Developer','Engineer','Manager','Specialist','Technician'])
years_at_company = st.number_input("Years at Company", min_value=0, max_value=10, value=5)
education_level = st.selectbox("Education Level", ["Bachelor", "High School", "Master", "PhD"])
monthly_salary = st.number_input("Monthly Salary", min_value=1000, max_value=15000, value=5000)
work_hours_per_week = st.number_input("Work Hours Per Week", min_value=20, max_value=80, value=40)
projects_handled = st.number_input("Projects Handled", min_value=0, max_value=50, value=5)
overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=50, value=5)
remote_work_frequency = st.selectbox("Remote Work Frequency", [0,25,50,75,100])
training_hours = st.number_input("Training Hours", min_value=0, max_value=150, value=10)
promotions = st.number_input("Promotions", min_value=0, max_value=5, value=1)
employee_satisfaction_score = st.slider("Employee Satisfaction Score", 1, 4, 2)

job_title_encoded = le_job.transform([job_title])[0]
education_encoded = le_edu.transform([education_level])[0]
remote_work_encoded = le_rwf.transform([remote_work_frequency])[0]
satisfaction_encoded = le_ess.transform([employee_satisfaction_score])[0]

input_features = np.array([[job_title_encoded,years_at_company,education_encoded, monthly_salary, work_hours_per_week, projects_handled, overtime_hours,
                            remote_work_encoded, training_hours, promotions,satisfaction_encoded]])

input_features_scaled = scaler.transform(input_features)

if st.button("Predict Performance"):
    prediction = model.predict(input_features_scaled)
    st.success(f"Predicted Performance Score: {prediction[0]}")

