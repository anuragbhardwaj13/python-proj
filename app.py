import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Set page config
st.set_page_config(page_title="Hospital Readmission Predictor", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoders = {}
    st.session_state.scaler = None

def load_and_train_model():
    # Load data
    df = pd.read_csv('hospital_readmissions_large.csv')
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Prepare features
    X, y, label_encoders, scaler = prepare_features(df_clean)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, label_encoders, scaler

def preprocess_data(df):
    # Handle missing values
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

def prepare_features(df):
    # Initialize encoders and scaler
    label_encoders = {}
    scaler = StandardScaler()
    
    # Encode categorical variables
    categorical_cols = ['gender', 'insurance_type', 'admission_type', 'diagnosis_code']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Prepare features and target
    feature_cols = ['age', 'gender', 'insurance_type', 'admission_type', 'diagnosis_code',
                   'length_of_stay', 'num_previous_admissions', 'num_comorbidities',
                   'emergency_visits_past_year']
    
    X = df[feature_cols]
    y = df['readmitted']
    
    # Scale features
    X = scaler.fit_transform(X)
    
    return X, y, label_encoders, scaler

def make_prediction(input_data):
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in ['gender', 'insurance_type', 'admission_type', 'diagnosis_code']:
        input_df[col] = st.session_state.label_encoders[col].transform(input_df[col])
    
    # Scale features
    input_scaled = st.session_state.scaler.transform(input_df)
    
    # Make prediction
    prediction = st.session_state.model.predict_proba(input_scaled)
    return prediction[0][1]  # Return probability of readmission

# Main app
def main():
    st.title("Hospital Readmission Prediction System")
    
    # Initialize or load model
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            model, label_encoders, scaler = load_and_train_model()
            st.session_state.model = model
            st.session_state.label_encoders = label_encoders
            st.session_state.scaler = scaler
        st.success("Model loaded successfully!")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=65)
            gender = st.selectbox("Gender", options=['M', 'F'])
            insurance = st.selectbox("Insurance Type", 
                                   options=['Medicare', 'Medicaid', 'Private', 'Self-pay'])
            admission = st.selectbox("Admission Type", 
                                   options=['Emergency', 'Elective', 'Urgent'])
        
        with col2:
            diagnosis = st.selectbox("Diagnosis Code", 
                                   options=['I25.10', 'I50.9', 'J44.9', 'E11.9', 'J18.9',
                                          'N17.9', 'K92.2', 'I63.9', 'I48.91', 'F32.9'])
            los = st.number_input("Length of Stay (days)", min_value=1, max_value=30, value=5)
            prev_admissions = st.number_input("Number of Previous Admissions", 
                                            min_value=0, max_value=10, value=0)
            comorbidities = st.number_input("Number of Comorbidities", 
                                          min_value=0, max_value=10, value=1)
            emergency_visits = st.number_input("Emergency Visits in Past Year", 
                                             min_value=0, max_value=10, value=0)
        
        submit_button = st.form_submit_button("Predict Readmission Risk")
    
    # Make prediction when form is submitted
    if submit_button:
        input_data = {
            'age': age,
            'gender': gender,
            'insurance_type': insurance,
            'admission_type': admission,
            'diagnosis_code': diagnosis,
            'length_of_stay': los,
            'num_previous_admissions': prev_admissions,
            'num_comorbidities': comorbidities,
            'emergency_visits_past_year': emergency_visits
        }
        
        risk_score = make_prediction(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Readmission Risk Score", f"{risk_score:.1%}")
        
        with col2:
            risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
            st.metric("Risk Level", risk_level)
        
        # Additional information
        if risk_score > 0.7:
            st.warning("⚠️ High risk of readmission. Consider additional follow-up care.")
        elif risk_score > 0.3:
            st.info("ℹ️ Moderate risk of readmission. Monitor patient's progress.")
        else:
            st.success("✅ Low risk of readmission. Continue standard care plan.")

if __name__ == "__main__":
    main()