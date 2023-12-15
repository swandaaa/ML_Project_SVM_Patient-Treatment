import os
import streamlit as st
import numpy as np
import joblib
import pandas as pd

current_directory = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_directory, 'SVM_model.pkl')
scaler_path = os.path.join(current_directory, 'scaler.pkl')

SVM_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Fungsi prediksi menggunakan model SVM
def patient_treatment_classification(haematocrit, haemoglobins, erythrocyte, leucocyte, thrombocyte, mch, mcv, age, sex):
    haematocrit = float(haematocrit)
    haemoglobins = float(haemoglobins)
    erythrocyte = float(erythrocyte)
    leucocyte = float(leucocyte)
    thrombocyte = float(thrombocyte)
    mch = float(mch)
    mcv = float(mcv)
    age = float(age)
    if sex == 'Laki-laki' :
        sex = 1
    else:
        sex = 0

    # Kumpulkan semua fitur dalam satu DataFrame
    features_df = pd.DataFrame({
        'HAEMATOCRIT': [haematocrit],
        'HAEMOGLOBINS': [haemoglobins],
        'ERYTHROCYTE': [erythrocyte],
        'LEUCOCYTE': [leucocyte],
        'THROMBOCYTE': [thrombocyte],
        'MCH': [mch],
        'MCV': [mcv],
        'AGE': [age],
        'SEX': [sex]
    })

    # Menampilkan tabel fitur sebelum di normalisasi
    st.write("Hasil Ekstraksi Fitur Sebelum Normalisasi")
    st.dataframe(features_df)

    # Menggunakan scaler untuk transformasi data
    normalized_data = loaded_scaler.transform(features_df)

    # Menampilkan tabel fitur setelah di normalisasi
    st.write("Hasil Ekstraksi Fitur Setelah Normalisasi")
    st.dataframe(pd.DataFrame(normalized_data, columns=features_df.columns))

    # Prediksi emosi menggunakan model SVM
    prediction = SVM_model.predict(normalized_data)

    
    return prediction[0]

def main():
    st.title('Aplikasi Klasifikasi Perawatan Pasien dengan SVM')

    # Input parameter haematocrit
    haematocrit_input = st.text_input("Masukkan nilai haematocrit:")
    # Input parameter haemoglobins
    haemoglobins_input = st.text_input("Masukkan nilai haemoglobins:")
    # Input parameter erythrocyte
    erythrocyte_input = st.text_input("Masukkan nilai erythrocyte:")
    # Input parameter leucocyte
    leucocyte_input = st.text_input("Masukkan nilai leucocyte:")
    # Input parameter thrombocyte
    thrombocyte_input = st.text_input("Masukkan nilai thrombocyte:")
    # Input parameter mch
    mch_input = st.text_input("Masukkan nilai mch:")
    # Input parameter mcv
    mcv_input = st.text_input("Masukkan nilai mcv:")
    # Input parameter age
    age_input = st.text_input("Masukkan nilai age:")
    # Input parameter sex
    sex_input = st.radio("Pilih jenis kelamin:", ('Laki-laki', 'Perempuan'))
    
    # Melakukan prediksi
    if st.button('Prediksi'):
        # Memanggil fungsi prediksi
        prediction = patient_treatment_classification(haematocrit_input, haemoglobins_input, erythrocyte_input, leucocyte_input, thrombocyte_input, mch_input, mcv_input, age_input, sex_input)


        st.write('Hasil Prediksi:')
        st.write(prediction)

if __name__ == "__main__":
    main()
