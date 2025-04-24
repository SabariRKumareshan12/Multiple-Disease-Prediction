import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

st.title("Multiple Disease Prediction :stethoscope:")

category=st.sidebar.radio("Select category", ["Home","Parkinson's Disease", "Kidney Disease", "Liver Disease"])

if category == "Parkinson's Disease":
    st.markdown("#### Parkinson's Disease Prediction")
    try:
        parkinson_model = pickle.load(open(r"E:/user data/Downloads/XGParkinson.pkl", 'rb'))
    except FileNotFoundError:
        st.error("File not found :astonished:")
        st.stop()

    # Input fields for Parkinson's disease prediction
    MDVP_Fo_Hz = st.number_input("Fundamental Frequency (MDVP:Fo(Hz))", min_value=0.0, value=0.0)
    MDVP_Fhi_Hz = st.number_input("Maximum Frequency (MDVP:Fhi(Hz))", min_value=0.0, value=0.0)
    MDVP_Flo_Hz = st.number_input("Minimum Frequency (MDVP:Flo(Hz))", min_value=0.0, value=0.0)
    MDVP_Jitter_percent = st.number_input("Jitter (MDVP:Jitter(%))", min_value=0.0, value=0.0,format="%.5f")
    MDVP_Jitter_Abs = st.number_input("Absolute Jitter (MDVP:Jitter(Abs))", min_value=0.0, value=0.0,format="%.5f")
    MDVP_RAP = st.number_input("Relative Average Perturbation (MDVP:RAP)", min_value=0.0, value=0.0,format="%.5f")
    MDVP_PPQ = st.number_input("Pitch Period Perturbation Quotient (MDVP:PPQ)", min_value=0.0, value=0.0,format="%.5f")
    Jitter_DDP = st.number_input("Degree of Derivative Perturbation (Jitter:DDP)", min_value=0.0, value=0.0,format="%.5f")
    MDVP_Shimmer = st.number_input("Shimmer (MDVP:Shimmer)", min_value=0.0, value=0.0,format="%.5f")
    MDVP_Shimmer_dB = st.number_input("Shimmer in dB (MDVP:Shimmer(dB))", min_value=0.0, value=0.0)
    Shimmer_APQ3 = st.number_input("Amplitude Perturbation Quotient (Shimmer:APQ3)", min_value=0.0, value=0.0,format="%.5f")
    Shimmer_APQ5 = st.number_input("Amplitude Perturbation Quotient (Shimmer:APQ5)", min_value=0.0, value=0.0,format="%.5f")
    MDVP_APQ = st.number_input("Amplitude Perturbation Quotient (MDVP:APQ)", min_value=0.0, value=0.0,format="%.5f")
    Shimmer_DDA = st.number_input("Difference of Average Amplitude (Shimmer:DDA)", min_value=0.0, value=0.0,format="%.5f")
    NHR = st.number_input("Noise-to-Harmonics Ratio (NHR)", min_value=0.0, value=0.0,format="%.5f")
    HNR = st.number_input("Harmonics-to-Noise Ratio (HNR)", min_value=0.0, value=0.0)
    RPDE = st.number_input("Recurrence Period Density Entropy (RPDE)", min_value=0.0, value=0.0,format="%.5f")
    DFA = st.number_input("Detrended Fluctuation Analysis (DFA)", min_value=0.0, value=0.0,format="%.5f")
    spread1 = st.number_input("Signal Spread 1 (spread1)", value=0.0,format="%.5f")
    spread2 = st.number_input("Signal Spread 2 (spread2)", value=0.0,format="%.5f")
    D2 = st.number_input("Correlation Dimension (D2)", min_value=0.0, value=0.0,format="%.5f")
    PPE = st.number_input("Pitch Period Entropy (PPE)", min_value=0.0, value=0.0,format="%.5f")

    input_features = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,RPDE, DFA, spread1, spread2, D2, PPE]])

    if st.button("Predict"):
        try:
            prediction = parkinson_model.predict(input_features)
            if prediction[0] == 1:
                st.success("You have Parkinson's disease :cry:")
            else:
                st.success("You do not have Parkinson's disease :grin:")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif category == "Kidney Disease":
    st.markdown("#### Kidney Disease Prediction")
    try:
        kidney_model = pickle.load(open(r"E:/user data/Downloads/XGKidney.pkl", 'rb'))
    except FileNotFoundError:
        st.error("File not found :astonished:")
        st.stop()

    Age = st.number_input("Age", min_value=1, max_value=120, value=30)
    Blood_Pressure = st.number_input("Blood Pressure", min_value=1, max_value=200, value=80)
    Specific_Gravity = st.number_input("Specific Gravity", min_value=1.0, max_value=1.03, value=1.02, format="%.2f")
    Albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4])  # Assuming Albumin is categorical (0-4)
    Sugar = st.selectbox("Sugar", [0, 1])  # Binary (0 or 1)
    Red_Blood_Cells = st.selectbox("Red Blood Cells", ["normal", "abnormal"])  # Categorical
    Pus_Cell = st.selectbox("Pus Cell", ["normal", "abnormal"])  # Categorical
    Pus_Cell_Clumps = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])  # Categorical
    Bacteria = st.selectbox("Bacteria", ["present", "notpresent"])  # Categorical
    Blood_Glucose_Random = st.number_input("Blood Glucose Random", min_value=0.0, value=0.0)
    Blood_Urea = st.number_input("Blood Urea", min_value=0.0, value=0.0)
    Serum_Creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=0.0)
    Sodium = st.number_input("Sodium", min_value=0.0, value=0.0)
    Potassium = st.number_input("Potassium", min_value=0.0, value=0.0)
    Hemoglobin = st.number_input("Hemoglobin", min_value=0.0, value=0.0)
    Packed_Cell_Volume = st.number_input("Packed Cell Volume", min_value=0.0, value=0.0)
    White_Blood_Cell_Count = st.number_input("White Blood Cell Count", min_value=0.0, value=0.0)
    Red_Blood_Cell_Count = st.number_input("Red Blood Cell Count", min_value=0.0, value=0.0)
    Hypertension = st.selectbox("Hypertension", ["yes", "no"])
    Diabetes_Mellitus = st.selectbox("Diabetes Mellitus", ["yes", "no"])
    Coronary_Artery_Disease = st.selectbox("Coronary Artery Disease", ["yes", "no"])
    Appetite = st.selectbox("Appetite", ["good", "poor"])
    Pedal_Edema = st.selectbox("Pedal Edema", ["yes", "no"])
    Anemia = st.selectbox("Anemia", ["yes", "no"])

    # Map Specific Gravity to encoded values (if required)
    specific_gravity_mapping = {1.005: 0, 1.01: 1, 1.015: 2}  # Example mapping
    Specific_Gravity = specific_gravity_mapping.get(Specific_Gravity, -1)  # Default to -1 if not mapped

    input_features = np.array([[Age,Blood_Pressure,Specific_Gravity,Albumin,Sugar,1 if Red_Blood_Cells == "abnormal" else 0,1 if Pus_Cell == "abnormal" else 0,
                                1 if Pus_Cell_Clumps == "present" else 0,1 if Bacteria == "present" else 0,Blood_Glucose_Random,Blood_Urea,
                                Serum_Creatinine,Sodium,Potassium,Hemoglobin,Packed_Cell_Volume,White_Blood_Cell_Count,Red_Blood_Cell_Count,
                                1 if Hypertension == "yes" else 0,1 if Diabetes_Mellitus == "yes" else 0,1 if Coronary_Artery_Disease == "yes" else 0,
                                1 if Appetite == "poor" else 0,1 if Pedal_Edema == "yes" else 0,1 if Anemia == "yes" else 0]]).astype(float)
    # Cleanse string columns (if necessary)
    for col in range(input_features.shape[1]):
        input_features[:, col] = [str(x).encode('utf-8').decode('utf-8') if isinstance(x, str) else x for x in input_features[:, col]]

    if st.button("Predict"):
        try:
            prediction = kidney_model.predict(input_features)
            if prediction[0] == 1:
                st.success("You have Kidney disease :cry:")
            else:
                st.success("You do not have Kidney disease :grin:")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif category == "Liver Disease":
    st.markdown("#### Liver Disease Prediction")

    try:
        liver_model = pickle.load(open("E:/user data/Downloads/XGBLiver.pkl", 'rb'))
    except FileNotFoundError:
        st.error("File not found :astonished:")
        st.stop()    
    
    Age= st.number_input("Age", min_value=1, max_value=120, value=30)
    Gender = st.selectbox("Gender", [1.0, 0.0], format_func=lambda x: "Female" if x == 1.0 else "Male")
    Total_Bilirubin= st.number_input("Total Bilirubin", min_value=0.0, value=0.0)
    Direct_Bilirubin= st.number_input("Direct Bilirubin", min_value=0.0, value=0.0)
    Alkaline_Phosphotase= st.number_input("Alkaline Phosphotase", min_value=0, value=0)
    Alamine_Aminotransferase= st.number_input("Alamine Aminotransferase", min_value=0, value=0)
    Aspartate_Aminotransferase= st.number_input("Aspartate Aminotransferase", min_value=0, value=0)
    Total_Proteins= st.number_input("Total Proteins", min_value=0.0, value=0.0)
    Albumin= st.number_input("Albumin", min_value=0.0, value=0.0)
    Albumin_and_Globulin_Ratio= st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=0.0)

    input_features = np.array([[Age,float(Gender),Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,
                                Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Proteins,
                                Albumin,Albumin_and_Globulin_Ratio]]).astype(float)
    # Cleanse string columns (if necessary)
    for col in range(input_features.shape[1]):
        input_features[:, col] = [str(x).encode('utf-8').decode('utf-8') if isinstance(x, str) else x for x in input_features[:, col]]
 
    if st.button("Predict"):
        try:
            prediction = liver_model.predict(input_features)
            if prediction[0] == 0:
                st.success("You do not have Liver disease :grin:")
            else:
                st.success("You have Liver disease :cry:")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif category == "Home":
    st.markdown("#### Welcome")
    st.markdown("This tool helps you predict the likelihood of various diseases based on your symptoms and health data.")
    st.markdown("##### 🧠 What you can do here:")
    st.markdown("""
    - 🔍 **Predict multiple diseases** including Kidney Disease, Liver Disease, and Parkinson's Disease
    - 📊 **Input your medical data** in a simple, user-friendly interface
    - 🤖 **Get instant predictions** powered by machine learning models""")

    st.markdown("Navigate through the sidebar to explore specific analyses")

st.markdown("Thank you :blush:")


