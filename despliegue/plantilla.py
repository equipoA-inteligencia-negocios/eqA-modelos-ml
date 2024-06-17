import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

def cargar_modelo(path):
    return joblib.load(path)

def app():
    st.title('Aplicación para Seleccionar Archivo Excel')
    
    # Definir los archivos Excel predefinidos
    excel_files = {
        "Buenaventura": "produccion/csv/BVN.csv",
        "Fortuna Silver Mines": "produccion/csv/FSM.csv",
        "SCCO": "produccion/csv/SCCO.csv"
    }

    models = {
        "Modelo híbrido": "produccion/models/modelo_hibrido.pkl"
        # Agregar los otros modelos
    }
    
    # Crear un listbox para seleccionar el archivo
    selected_file_name = st.selectbox('Selecciona un archivo', list(excel_files.keys()))
    selected_model_name = st.selectbox('Selecciona un modelo', list(models.keys()))
    begin_date = st.date_input('Selecciona una fecha de inicio')
    end_date = st.date_input('Selecciona una fecha de fin')
    
    if selected_file_name:
        if selected_model_name:
            selected_model_path = models[selected_model_name]
            model = cargar_modelo(selected_model_path)
            selected_file_path = excel_files[selected_file_name]
            df = pd.read_csv(selected_file_path)
            st.write(f"Mostrando datos del archivo: {selected_file_name}")
            st.dataframe(df)

            # Aquí puedes agregar más funciones en base al archivo seleccionado
            # Por ejemplo, calcular estadísticas, generar gráficos, etc.

            st.write("Descripción del DataFrame:")
            st.write(df.describe())