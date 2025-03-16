# Employee Performance Prediction Using Machine Learning
### Project Overview:
The Employee Performance Prediction project aims to predict the performance scores of employees based on various workplace and personal factors. By leveraging machine learning models, the project identifies key factors influencing employee performance and provides actionable insights for HR teams to make data-driven decisions. This predictive system helps improve workforce management, identify high-performing employees, and provide early intervention for underperformers.
### Objective
- Build a machine learning model to accurately predict employee performance.
- Analyze the relationship between various employee attributes (e.g., work hours, salary, education) and performance.
- Deploy an interactive web application using Streamlit for easy model access and predictions.
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
#### Data Loading:
- Import the dataset containing employee information.
- Verify data integrity by checking for missing values, duplicate records, and inconsistencies.
- Load additional resources such as label encoders and scalers for consistent preprocessing.
#### Data Preprocessing:
**Handle Missing Values and Outliers**
  - Impute missing values using appropriate strategies (e.g., mean/median for numerical features, mode for categorical).
  - Detect and manage outliers using the **IQR method** or **Z-score**.
**Encode Categorical Variables**
  - Apply **Label Encoding** for categorical columns (e.g., Job Title, Education Level, Remote Work Frequency).
  - Ensure consistent mapping of labels during training and deployment.
**Standardize Numerical Features**
  - Use **StandardScaler** to scale numerical variables (e.g., monthly salary, work hours, training hours) for consistent model performance.
#### Exploratory Data Analysis (EDA):
**Data Summary**
  - Perform statistical analysis to understand the distribution of features (mean, median, standard deviation).
  - Analyze the class distribution of the **Performance Score**.
**Feature Relationships**
  - Visualize correlations between input features using a **heatmap**.
  - Explore relationships between performance score and key attributes (e.g., work hours, salary, satisfaction).
**Class Imbalance Analysis**
  - Identify the imbalance in the target variable and assess its impact on model training.
  - Evaluate the necessity of oversampling or undersampling techniques.
#### Model Development:
**Model Selection**
  - Implement and evaluate the following machine learning algorithms:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - GaussianNB
    - Support Vector Classifier (SVC)
    - Decision Tree Classifier
    - Random Forest Classifier
    - Gradient Boosting (GB), AdaBoost, and XGBoost
**Hyperparameter Tuning**
  - Use GridSearchCV to optimize hyperparameters for each model.
  - Perform 5-fold cross-validation to ensure generalizability and reduce overfitting.
#### Model Evaluation:
  - Evaluate model performance using:
    - Accuracy
    - Precision, Recall, and F1-score (for class-specific insights)
    - Confusion Matrix for a detailed error analysis
#### Model Deployment:
**Model Saving**
  - Save the best-performing model and preprocessing objects (scalers, label encoders) using joblib for future use.
**Streamlit Web Application**
  - Build an interactive Streamlit application.
### Technologies Used
Python: Core programming language
Libraries:
  - Machine Learning: scikit-learn, xgboost, imblearn
  - Data Analysis: pandas, numpy
  - Visualization: matplotlib, seaborn

Web Deployment: Streamlit
Tools: Jupyter Notebook, Joblib (model persistence)
### Outcomes
Developed a machine learning model with >99% training accuracy and optimized generalization performance.
Built a user-friendly Streamlit app where HR professionals can input employee data and receive performance predictions.
