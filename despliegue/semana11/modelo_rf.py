import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Configuración de estilo de Seaborn para gráficos
sns.set(style="whitegrid")

# Función para descargar y procesar datos de Yahoo Finance
def descargar_datos(ticker, fecha_inicio, fecha_fin):
    data = yf.download(ticker, start=fecha_inicio, end=fecha_fin)
    data.drop(['Adj Close'], axis=1, inplace=True)
    data.columns += f"_{ticker}"
    return data

# Función para preprocesar datos
def preprocesar_datos(data, ticker):
    data['Tomorrow'] = data[f'Close_{ticker}'].shift(-1)
    data['Target'] = (data['Tomorrow'] > data[f'Close_{ticker}']).astype(int)
    data = data.dropna(subset=['Tomorrow', 'Target'])
    return data

# Función para entrenar el modelo
def entrenar_modelo(data, ticker):
    predictors = [f'Close_{ticker}', f'Volume_{ticker}', f'Open_{ticker}', f'High_{ticker}', f'Low_{ticker}']
    train = data.iloc[:-100]
    test = data.iloc[-100:]

    model = RandomForestClassifier(n_estimators=200, max_depth=3, max_features=4, random_state=42)
    model.fit(train[predictors], train['Target'])
    
    return model, predictors, test

# Evaluación del modelo
def evaluate_model(model, test, predictors):
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    precision = precision_score(test['Target'], preds)
    return precision, preds

def app():
    st.title("Modelo de Predicción de Precios de Acciones")
    
    st.write("""
    ### Bienvenido a la aplicación de predicción de precios de acciones. 
    Esta aplicación utiliza un modelo de Random Forest para predecir los precios de cierre de acciones y determinar la tendencia del mercado.
    Por favor, introduzca los parámetros necesarios para comenzar.
    """)

    # Parámetros de entrada del usuario
    st.subheader("Configuración del Modelo")
    ticker = st.selectbox("Seleccione la cotización bursátil", ["BVN", "FSM", "SCCO"])
    fecha_inicio = st.date_input("Fecha de Inicio", pd.to_datetime("2022-01-01"))
    fecha_fin = st.date_input("Fecha de Fin", pd.to_datetime("2023-01-01"))

    if st.button("Cargar Datos"):
        # Descargar y preprocesar datos
        data_load_state = st.text("Cargando datos...")
        datos = descargar_datos(ticker, fecha_inicio, fecha_fin)
        data_load_state.text("Datos cargados con éxito!")

        # Mostrar datos
        st.subheader(f"Datos de {ticker} desde {fecha_inicio} hasta {fecha_fin}")
        st.dataframe(datos)

        # Preprocesamiento de datos
        st.subheader("Preprocesamiento de Datos")
        st.write("""
        Se crea una columna 'Tomorrow' que contiene el precio de cierre del día siguiente y una columna 'Target' que indica si el precio subió o bajó.
        Estos datos son necesarios para entrenar y evaluar el modelo de predicción.
        """)
        datos = preprocesar_datos(datos, ticker)

        # Entrenar modelo
        st.subheader("Entrenamiento del Modelo")
        st.write("""
        El modelo se entrena utilizando un Random Forest Classifier. 
        Este modelo de aprendizaje automático es eficaz para la predicción de series temporales debido a su capacidad para manejar grandes cantidades de datos y múltiples características.
        """)
        modelo, columnas, test = entrenar_modelo(datos, ticker)

        # Realizar predicciones
        st.subheader("Predicciones del Modelo")
        st.write("""
        Después de entrenar el modelo, se utilizan los datos de prueba para realizar predicciones sobre la tendencia de los precios de cierre para los días siguientes.
        """)
        precision, preds = evaluate_model(modelo, test, columnas)
        test['Predictions'] = preds
        test['Prediction Label'] = test['Predictions'].apply(lambda x: 'Subida' if x == 1 else 'Bajada')

        # Mostrar evaluación del modelo
        st.write(f"**Precisión del Modelo:** {precision:.4f}")

        # Mostrar predicciones
        st.write("**Predicciones realizadas por el modelo junto con los precios reales:**")
        st.dataframe(test[[f'Close_{ticker}', 'Tomorrow', 'Target', 'Predictions', 'Prediction Label']])

        # Mostrar recomendación
        st.subheader("Recomendación")
        recomendacion = "Comprar" if test['Predictions'].iloc[-1] == 1 else "Vender"
        st.write(f"**Recomendación para mañana:** **{recomendacion}**")

        # Visualización de datos y predicciones
        st.subheader("Visualización de Precios Reales y Predichos")
        st.write("A continuación se muestra un gráfico comparando los precios reales de cierre con las predicciones del modelo.")

        plt.figure(figsize=(14, 7))
        plt.plot(datos.index, datos[f'Close_{ticker}'], label='Precio Real', color='blue')
        plt.plot(test.index, test['Tomorrow'], label='Precio Real de Mañana', linestyle='--', color='green')
        plt.plot(test.index, test['Predictions'], label='Predicción', linestyle='--', color='red')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre')
        plt.title(f"Precios Reales y Predicciones para {ticker}")
        plt.legend()
        st.pyplot(plt)

        st.subheader("Media Móvil de los Precios Reales")
        st.write("El gráfico siguiente muestra la media móvil de 20 días de los precios de cierre.")
        datos['MA20'] = datos[f'Close_{ticker}'].rolling(window=20).mean()
        
        plt.figure(figsize=(14, 7))
        plt.plot(datos.index, datos[f'Close_{ticker}'], label='Precio Real', color='blue')
        plt.plot(datos.index, datos['MA20'], label='Media Móvil 20 Días', linestyle='--', color='orange')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre')
        plt.title(f"Media Móvil de 20 Días para {ticker}")
        plt.legend()
        st.pyplot(plt)

