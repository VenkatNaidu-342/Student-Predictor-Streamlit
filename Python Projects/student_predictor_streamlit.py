import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample Data
data = {
    'Hours_Studied': [2, 5, 1, 4, 6, 1.5, 3.5, 3, 5.5, 2.5],
    'Attendance': [60, 85, 45, 70, 90, 50, 75, 65, 88, 55],
    'Previous_Score': [55, 75, 40, 65, 85, 45, 70, 60, 80, 50],
    'Result': [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)

# Train ML Model
X = df[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎓 Student Performance Predictor")
st.write("Enter the details below:")

hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
score = st.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict"):
    input_data = np.array([[hours, attendance, score]])
    prediction = model.predict(input_data)
    result = "Pass ✅" if prediction[0] == 1 else "Fail ❌"
    st.success(f"The student will: **{result}**")