import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from despliegue.adding_data import calculate_indicators_mining_company, mostrar_correlacion
from despliegue.ltsm import ltsm_final

def cargar_modelo(path):
    return joblib.load(path)

def app():
    st.title('Aplicación para Seleccionar Archivo Excel')
    
    # Definir los archivos Excel predefinidos
    csv_files = {
        "Buenaventura": "BVN",
        "Fortuna Silver Mines": "FSM",
        "SCCO": "SCCO"
    }

    models = {
        "LTSM": "ltsm",
        "SVM": "svm",
        "Modelo híbrido": "mh",
        "Random Forest": "rf"
    }
    
    # Crear un listbox para seleccionar el archivo
    selected_file_name = st.selectbox('Selecciona un archivo', list(csv_files.keys()))
    selected_model_name = st.selectbox('Selecciona un modelo', list(models.keys()))
    min_date = datetime(2018, 1, 1)
    max_date = datetime(2023, 12, 31)
        
    begin_date = st.date_input('Selecciona una fecha de inicio' ,min_value=min_date, max_value=max_date)
    end_date = st.date_input('Selecciona una fecha de fin', min_value=min_date, max_value=max_date)
    
    if selected_file_name:
        if selected_model_name:
            selected_model = models[selected_model_name]
            selected_file_path = csv_files[selected_file_name]
            df_main = yf.download(selected_file_path, start=begin_date, end=end_date)
            df_main.columns += "_" + selected_file_path
            st.dataframe(df_main)
            st.write("Descripción del DataFrame:")
            st.write(df_main.describe())
            df_main = calculate_indicators_mining_company(df_main, selected_file_path)
            st.write('Dataframe con valores añadidos:')
            st.dataframe(df_main)
            if selected_model == 'ltsm':
                st.write('Modelo LTSM')
                st.pyplot(mostrar_correlacion(df_main))
                df_fsm = ltsm_final(df_main, begin_date, end_date)
                st.dataframe(df_fsm)
                # Colocar modelo ltsm
            elif selected_model == 'svm':
                st.write('Modelo SVM')
                # Colocar modelo svm
            elif selected_model == 'mh':
                st.write('Modelo Híbrido')
                # Colocar modelo híbrido
            elif selected_model == 'rf':
                st.write('Modelo Random Forest')
                # Colocar modelo random forest