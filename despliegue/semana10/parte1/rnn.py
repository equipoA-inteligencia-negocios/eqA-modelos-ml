from despliegue.semana10.parte1.preprocesamiento import preprocesar
import streamlit as st
import pandas as pd
from despliegue.semana10.parte1.rnn_template import build, train, evaluate, predict_test
from despliegue.semana10.parte1.lstm_template import split_data, predict_next_day_price

def app():
    st.title("Modelo de Predicción de IGBVL")
    
    st.write("""
    ### Bienvenido a la aplicación de predicción de IGBVL. 
    Esta aplicación utiliza un modelo RNN para predecir el IGBVL.
    """)
    
    df = pd.read_csv('despliegue/semana10/parte1/Datos.csv')
    
    st.write("""
    Se tienen los siguientes datos:
    """)
    
    st.dataframe(df)
    
    df, scaler_y = preprocesar(df)

    seq_length_rnn = 60
    
    X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn, y_test_indexes_rnn = split_data(df, seq_length_rnn)
    
    model_rnn = build(seq_length_rnn, X_train_rnn)

    st.subheader("Entrenamiento del Modelo")
    st.write("Entrenando el modelo...")
    train(model_rnn, X_train_rnn, y_train_rnn)
    st.write("Modelo entrenado!")

    st.subheader("Evaluación del modelo")
    evaluate(model_rnn, X_test_rnn, y_test_rnn)

    st.subheader("Predicciones en el conjunto de prueba")
    predict_test(model_rnn, X_test_rnn, y_test_rnn, y_test_indexes_rnn, scaler_y)

    st.subheader("Predicción del IGBVL día siguiente")
    st.write(f'Se espera que el IGBVL del día siguiente sea de {predict_next_day_price(model_rnn, df, seq_length_rnn, scaler_y):.4f}')