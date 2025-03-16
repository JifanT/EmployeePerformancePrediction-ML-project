# Employee Performance Prediction
## Machine Learning Project
### Objective
This analysis aims to develop a predictive model that evaluates how different factors such as compensation, work environment, and employee growth opportunities correlate with an employee’s performance score by leveraging machine learning techniques.
### Understanding The Problem
####  Problem Definition:
Employee performance is a critical factor in organizational success. Various elements, such as salary, team size, and education level, influence an employee's productivity and overall performance. However, identifying the most impactful factors and quantifying their influence remains a challenge.
#### Importance of the Analysis:
This analysis helps optimize workforce productivity by identifying key performance drivers. It provides insights into how salary, training, and workplace factors impact employee satisfaction and retention. By leveraging data-driven insights, HR can make fair decisions on promotions, rewards, and career growth. Additionally, predictive workforce planning enables organizations to invest in high-potential employees and drive long-term success.
### Dataset Overview
#### Summary:
This dataset contains 100,000 rows of data capturing key aspects of employee performance, productivity, and demographics in a corporate environment. It includes details related to the employee's job, work habits, education, performance, and satisfaction. The dataset is designed for various purposes such as HR analytics, employee churn prediction, productivity analysis, and performance evaluation.
#### Source:
The dataset used in this analysis has been sourced from Kaggle.
https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
#### Columns:
- Employee_ID: Unique identifier for each employee.
- Department: The department in which the employee works (e.g., Sales, HR, IT).
- Gender: Gender of the employee (Male, Female, Other).
- Age: Employee's age (between 22 and 60).
- Job_Title: The role held by the employee (e.g., Manager, Analyst, Developer).
- Hire_Date: The date the employee was hired.
- Years_At_Company: The number of years the employee has been working for the company.
- Education_Level: Highest educational qualification (High School, Bachelor, Master, PhD).
- Performance_Score: Employee's performance rating (1 to 5 scale).
- Monthly_Salary: The employee's monthly salary in USD, correlated with job title and performance score.
- Work_Hours_Per_Week: Number of hours worked per week.
- Projects_Handled: Total number of projects handled by the employee.
- Overtime_Hours: Total overtime hours worked in the last year.
- Sick_Days: Number of sick days taken by the employee.
- Remote_Work_Frequency: Percentage of time worked remotely (0%, 25%, 50%, 75%, 100%).
- Team_Size: Number of people in the employee's team.
- Training_Hours: Number of hours spent in training.
- Promotions: Number of promotions received during their tenure.
- Employee_Satisfaction_Score: Employee satisfaction rating (1.0 to 5.0 scale).
- Resigned: Boolean value indicating if the employee has resigned.
#### Target Feature:
- Performance_Score: Employee's performance rating (1 to 5 scale).
#### Dataset Dimension:
- Rows - 100000
- Columns - 20
### Methodology
- Data Acquisition
- Data Preprocess & Data Transformation
- Exploratory Data Analysis (EDA)
- Model Training & Evaluation
- Model Deployment
