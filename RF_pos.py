###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('RF_pos.pkl')
scaler = joblib.load('scaler.pkl') 

# Define feature names
feature_names = ["Level_of_Education", "Tumor_Size_2_Years_after_Surgery", "Tumor_Grade", "Lymph_Node_Metastasis", "Numbe_of_Lymph_Nodes", "Marital_Status_Unmarried", "Marital_Status_Married", "Marital_Status_Divorced"]

# Streamlit user interface
st.title("Breast Cancer Recurrence Predictor")

# Level_of_Educations
Level_of_Education = st.selectbox("Level of Education:", options=list(Level_of_Education_options.keys()), format_func=lambda x: Level_of_Education_options[x])

# Tumor_Size_2_Years_after_Surgery
Tumor_Size_2_Years_after_Surgery = st.number_input("Tumor Size 2 Years after Surgery(mm):", min_value=0, max_value=100, value=50)

# Tumor_Grade
Tumor_Grade = st.selectbox("Tumor Grade:", options=list(Tumor_Grade_options.keys()), format_func=lambda x: Tumor_Grade_options[x])

# Lymph_Node_Metastasis
Lymph_Node_Metastasis = st.selectbox("Lymph Node Metastasis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Numbe_of_Lymph_Nodes
Numbe_of_Lymph_Nodes = st.number_input("Numbe of Lymph Nodes:", min_value=0, max_value=50, value=25)

# Marital_Status_Unmarried
Marital_Status_Unmarried = st.selectbox("Marital Status Unmarried:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Married
Marital_Status_Married = st.selectbox("Marital Status Married:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Divorced
Marital_Status_Divorced = st.selectbox("Marital Status Divorced:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Process inputs and make predictions
feature_values = [ Level_of_Education, Tumor_Size_2_Years_after_Surgery, Tumor_Grade, Lymph_Node_Metastasis, Numbe_of_Lymph_Nodes, Marital_Status_Unmarried, Marital_Status_Married, Marital_Status_Divorced]
features = np.array([feature_values])

if st.button("Predict"):    
    # 标准化特征
    standardized_features = scaler.transform(features)

    # Predict class and probabilities    
    predicted_class = model.predict(standardized_features)[0]   
    predicted_proba = model.predict_proba(standardized_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(1: Disease, 0: No Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
         advice = (            
                f"According to our model, you have a high risk of breast cancer recurrence. "            
                f"The model predicts that your probability of having breast cancer recurrence is {probability:.1f}%. "            
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."        
          )    
    else:        
         advice = (           
                f"According to our model, you have a low risk of breast cancer recurrence. "            
                f"The model predicts that your probability of not having breast cancer recurrence is {probability:.1f}%. "            
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."        
          )    
    st.write(advice)

# SHAP Explanation    
    st.subheader("SHAP Force Plot Explanation")    
    explainer_shap = shap.TreeExplainer(model)    
    shap_values = explainer_shap.shap_values(pd.DataFrame(standardized_features, columns=feature_names))    
# 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)
# Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], original_feature_values, matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], original_feature_values, matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')