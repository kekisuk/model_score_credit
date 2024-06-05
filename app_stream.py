import streamlit as st
from joblib import load
import numpy as np


model = load('model.jbl')

st.title('Modelo de crédito')

feature1 = st.number_input('Digite a taxa de débito em relação a renda:', format="%.2f")
feature2 = st.number_input('Digite a taxa de mantimentos em relação a renda:', format="%.2f")
feature3 = st.number_input('Digite a taxa de renda:', format="%.2f")


if st.button('Predizer'):

    input_features = np.array([feature1, feature2, feature3]).reshape(1, -1)
    
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    if prediction[0] == 1:
        st.write(f'Crédito aprovado.')
    else:
        st.write(f'Crédito em análise. Forneça mais dados.')
    st.write(f'Probabilidades de aprovação:{prediction_proba[:, 1]}')

