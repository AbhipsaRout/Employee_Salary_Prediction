import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("GradientBoosting_model.pkl")

st.title("🧑‍💻Employee Income Prediction App")


# Label encoding dictionaries (must match training)
workclass_dict = {
    'Private': 0, 'Self-emp-not-inc': 1, 'Local-gov': 2, 'State-gov': 3,
    'Federal-gov': 4, 'Self-emp-inc': 5, '?': 6
}
marital_status_dict = {
    'Never-married': 0, 'Married-civ-spouse': 1, 'Divorced': 2,
    'Widowed': 3, '?': 4
}
occupation_dict = {
    'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3,
    'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6,
    'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9,
    'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12,
    'Armed-Forces': 13, '?': 14
}
relationship_dict = {
    'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3,
    'Other-relative': 4, 'Unmarried': 5
}
race_dict = {
    'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2,
    'Amer-Indian-Eskimo': 3, 'Other': 4
}
gender_dict = {'Male': 0, 'Female': 1}
native_country_dict = {
    'United-States': 0, 'Mexico': 1, 'Philippines': 2,
    'Germany': 3, 'Canada': 4, 'India': 5, '?': 6
}

# Collect user inputs
age = st.number_input("Age", min_value=17, max_value=90)
workclass = st.selectbox("Workclass", list(workclass_dict.keys()))
fnlwgt = st.number_input("Fnlwgt", min_value=0)
educational_num = st.slider("Educational Number", min_value=1, max_value=16)
marital_status = st.selectbox("Marital Status", list(marital_status_dict.keys()))
occupation = st.selectbox("Occupation", list(occupation_dict.keys()))
relationship = st.selectbox("Relationship", list(relationship_dict.keys()))
race = st.selectbox("Race", list(race_dict.keys()))
gender = st.selectbox("Gender", list(gender_dict.keys()))
capital_gain = st.number_input("Capital Gain", min_value=0)
capital_loss = st.number_input("Capital Loss", min_value=0)
hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99)
native_country = st.selectbox("Native Country", list(native_country_dict.keys()))

# Encode categorical inputs
workclass_encoded = workclass_dict[workclass]
marital_status_encoded = marital_status_dict[marital_status]
occupation_encoded = occupation_dict[occupation]
relationship_encoded = relationship_dict[relationship]
race_encoded = race_dict[race]
gender_encoded = gender_dict[gender]
native_country_encoded = native_country_dict[native_country]

# Final feature vector
input_data = np.array([[
    age,
    workclass_encoded,
    fnlwgt,
    educational_num,
    marital_status_encoded,
    occupation_encoded,
    relationship_encoded,
    race_encoded,
    gender_encoded,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country_encoded
]])

# Predict
if st.button("Predict Income Category"):
    prediction = model.predict(input_data)
    income = prediction[0]
    st.success(f"💰 Predicted Income Category: {income}")
