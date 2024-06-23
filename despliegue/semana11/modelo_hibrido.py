import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

def cargar_modelo():
    return joblib.load('produccion/models/modelo_hibrido.pkl')

def app():
    st.title("Modelo Híbrido")
    modelo = cargar_modelo()

    # Cargar datos para predicción
    uploaded_file = st.file_uploader("Ejemplo, cambiar", type=['csv'])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        
        # Supongamos que necesitas escalar los datos de entrada
        scaler = joblib.load('produccion/models/scaler.pkl')
        input_data_scaled = scaler.transform(input_data)

        # Predicciones
        predictions = modelo.predict(input_data_scaled)
        
        st.subheader('Resultados de la Predicción')
        st.write(predictions)

        # Gráficos de las predicciones vs los datos reales
        st.subheader('Gráficos de Predicción vs Datos Reales')
        fig, ax = plt.subplots()
        ax.plot(input_data.index, predictions, label='Predicciones')
        ax.plot(input_data.index, input_data['Close'], label='Datos Reales')
        ax.legend()
        st.pyplot(fig)

        # Evaluar el modelo (usando datos reales si están disponibles)
        if 'Close' in input_data.columns:
            real_data = input_data['Close']
            mape = mean_absolute_percentage_error(real_data, predictions)
            mse = mean_squared_error(real_data, predictions)
            r2 = r2_score(real_data, predictions)

            st.subheader('Evaluación del Modelo')
            st.write(f"MAPE: {mape:.4f}")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"R2: {r2:.4f}")