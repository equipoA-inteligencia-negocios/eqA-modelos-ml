from despliegue.semana10.parte1.preprocesamiento import preprocesar
import streamlit as st
import pandas as pd
from despliegue.semana10.parte1.lstm_template import split_data, build, train, evaluate, predict_test, plot_real_prices_full, predict_next_day_price

def app():
    st.title("Modelo de Predicción de IGBVL")
    
    st.write("""
    ### Bienvenido a la aplicación de predicción de IGBVL. 
    Esta aplicación utiliza un modelo LSTM para predecir el IGBVL.
    """)
    
    df = pd.read_csv('despliegue/semana10/parte1/Datos.csv')
    
    st.write("""
    Se tienen los siguientes datos:
    """)
    
    st.dataframe(df)
    
    df, scaler_y = preprocesar(df)
    
    st.subheader("Elección el modelo")
    model_type = st.selectbox("Seleccione el modelo", ["LSTMv1", "LSTMv2"])
    
    if (model_type == "LSTMv1"):
        model_type_prefix = "V1"
        epochs = 50
        batch_size = 32
        validation_split = 0.2
    else:
        model_type_prefix = "V2"
        epochs = 100
        batch_size = 64
        validation_split = 0.3
        
    st.write(f"# Modelo LSTM {model_type_prefix}")
    st.write(f"Para el entrenamiento de este modelo se tomaron en cuenta los siguientes hiperparámetros: \n - Épocas: {epochs} \n - Tamaño de lote: {batch_size} \n - División de validación: {validation_split}")
    
    seq_length_lstm = 60
    
    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, y_test_indexes_lstm = split_data(df, seq_length_lstm)
    
    model_lstm = build(seq_length_lstm, X_train_lstm)
    
    st.subheader("Entrenamiento del Modelo")
    st.write("""
        El modelo se entrena utilizando un LSTM. 
        Este modelo de aprendizaje automático es eficaz para la predicción de series temporales debido a su capacidad para manejar secuencias de datos.
    """)
    st.write("Entrenando el modelo...")
    train(model_lstm, X_train_lstm, y_train_lstm, epochs, batch_size, validation_split)
    st.write("Modelo entrenado!")
    
    st.subheader("Predicciones en el conjunto de prueba")
    mape, r2 = predict_test(model_lstm, X_test_lstm, y_test_lstm, y_test_indexes_lstm, scaler_y, model_type_prefix)
    
    st.subheader("Evaluación del modelo")
    evaluate(model_lstm, X_test_lstm, y_test_lstm, mape, r2)
    
    st.subheader("Predicciones en el conjunto de datos completo")
    df_predicciones = plot_real_prices_full(seq_length_lstm, df, model_lstm, scaler_y, model_type_prefix)
    st.subheader("Dataframe de predicciones en el conjunto de datos completo")
    st.dataframe(df_predicciones)
    
    st.subheader("Predicción del IGBVL día siguiente")
    st.write(f'Se espera que el IGBVL del día siguiente sea de {predict_next_day_price(model_lstm, df, seq_length_lstm, scaler_y):.4f}')