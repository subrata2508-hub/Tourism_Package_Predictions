import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Define the repository ID and filename for the model on Hugging Face
MODEL_REPO_ID = "subrata2508/Tourism-Package-Prediction" # Replace with your actual model repo ID
MODEL_FILENAME = "best_machine_failure_model_v1.joblib" # Replace with the actual filename of your saved model

# Download and load the model
@st.cache_resource
def load_best_model(repo_id, filename):
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

model = load_best_model(MODEL_REPO_ID, MODEL_FILENAME)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package.
Please enter the customer details below to get a prediction.
""")

if model is not None:
    # User input fields based on your dataset description
    st.header("Customer Details")

    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Small Business"]) # Add other occupations if present in data
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    Designation = st.text_input("Designation") # Consider making this a selectbox if categories are limited
    MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)

    st.header("Customer Interaction Data")
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]) # Add other pitched products
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)

    # Assemble input into DataFrame - Ensure column names match training data
    input_data = pd.DataFrame([{
        'TypeofContact': TypeofContact,
        'Age': Age,
        'CityTier': CityTier,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': Passport,
        'OwnCar': OwnCar,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'ProductPitched': ProductPitched,
        'NumberOfFollowups': NumberOfFollowups,
        'DurationOfPitch': DurationOfPitch
    }])


    if st.button("Predict Purchase"):
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[:, 1][0] # Probability of the positive class (purchase)

            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"The model predicts the customer is likely to **purchase** the package.")
            else:
                st.info(f"The model predicts the customer is **not likely** to purchase the package.")

            st.write(f"Predicted probability of purchase: **{prediction_proba:.2f}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure the input values are correct and the model is loaded properly.")

else:
    st.warning("Model not loaded. Please ensure the model is available on Hugging Face.")
