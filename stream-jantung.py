import pickle
import numpy as np
import streamlit as st

# load saved model
model = pickle.load(open('jantung.sav', 'rb'))

# title of the web app
st.title('Prediksi Penyakit Jantung')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Umur')
with col2:
    sex = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
with col3:
    cp = st.number_input('Jenis Nyeri Dada')
with col1:
    trestbps = st.number_input('Tekanan Darah')
with col2:
    chol = st.number_input('Nilai Kolesterol')
with col3:
    fbs = st.number_input('Gula Darah')
with col1:
    restech = st.number_input('Hasil Elektrokardiografi')
with col2:
    thalach = st.number_input('Detak Jantung Maksimum')
with col3:
    exang = st.number_input('Induksi Angina')
with col1:
    oldpeak = st.number_input('ST Depression')
with col2:
    slope = st.number_input('Slope')
with col3:
    ca = st.number_input('Nilai CA')
with col1:
    thal = st.number_input('Nilai Thal')

# Map the selected gender to numeric value
sex_numeric = 1 if sex == 'Laki-laki' else 0

# code for prediction
heart_diagnosis = ''

# create the prediction button
if st.button('Prediksi Penyakit Jantung'):
    heart_prediction = model.predict([[age, sex_numeric, cp, trestbps, chol, fbs, restech, thalach, exang, oldpeak, slope, ca, thal]])

    if heart_prediction[0] == 1: 
        heart_diagnosis = 'Pasien Terkena Penyakit Jantung'
    else:
        heart_diagnosis = 'Pasien Tidak Terkena Penyakit Jantung'
    
    st.success(heart_diagnosis)
