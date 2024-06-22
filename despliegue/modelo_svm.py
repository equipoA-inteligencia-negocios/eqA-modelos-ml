# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVR
from sklearn import metrics
from pandas_datareader import data as pdr
import yfinance as yfin
import plotly.express as px

def app():
    st.title('Modelo SVR para Predicción de Precios de Mineras')
    st.subheader('Seleccione los parámetros para la predicción')

    plt.style.use('fivethirtyeight')

    yfin.pdr_override()

    st.subheader("Obtener datos de Yahoo Finance")
    start = st.date_input('Inicio', value=pd.to_datetime('2014-01-01'))
    end = st.date_input('Fin', value=pd.to_datetime('2014-01-28'))
    stock_option = st.selectbox('Seleccione la cotización bursátil', ['BVN', 'FSM', 'SCCO'])

    df = pdr.get_data_yahoo(stock_option, start, end)
    st.write(df)
    st.subheader("Detalles de los datos")
    st.write(df.describe())
    st.write("Número de filas y columnas:")
    st.write(df.shape)

    # Obtener los datos hasta el penúltimo día
    df_train = df.head(len(df) - 1)
    df_test = df.tail(1)
    
    days = [[int(day.strftime('%d'))] for day in df_train.index]
    adj_close_prices = df_train['Adj Close'].tolist()

    # Crear y entrenar los modelos SVR
    lin_svr = SVR(kernel='linear', C=1000.0)
    lin_svr.fit(days, adj_close_prices)

    poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
    poly_svr.fit(days, adj_close_prices)

    rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
    rbf_svr.fit(days, adj_close_prices)

    # Predicción para el día siguiente
    prediction_day = [[int(df.index[-1].strftime('%d')) + 1]]
    rbf_prediction = rbf_svr.predict(prediction_day)[0]
    lin_prediction = lin_svr.predict(prediction_day)[0]
    poly_prediction = poly_svr.predict(prediction_day)[0]

    st.subheader('Predicciones para el día siguiente')
    st.write(f'Predicción RBF SVR: {rbf_prediction}')
    st.write(f'Predicción Lineal SVR: {lin_prediction}')
    st.write(f'Predicción Polinomial SVR: {poly_prediction}')

    fig = plt.figure(figsize=(16, 8))
    plt.scatter(days, adj_close_prices, color='red', label='Datos')
    plt.plot(days, rbf_svr.predict(days), color='green', label='Modelo RBF')
    plt.plot(days, poly_svr.predict(days), color='orange', label='Modelo Polinomial')
    plt.plot(days, lin_svr.predict(days), color='blue', label='Modelo Lineal')
    plt.legend()
    st.pyplot(fig)

    # Evaluación del modelo
    actual_values = [df_test['Adj Close'].values[0]]
    predicted_values = [rbf_prediction]

    MAE = metrics.mean_absolute_error(actual_values, predicted_values)
    MSE = metrics.mean_squared_error(actual_values, predicted_values)
    RMSE = np.sqrt(metrics.mean_squared_error(actual_values, predicted_values))

    metricas = pd.DataFrame({
        'Métrica': ['Error Absoluto Medio', 'Error Cuadrático Medio', 'Raíz del Error Cuadrático Medio'],
        'Valor': [MAE, MSE, RMSE]
    })

    st.subheader('Métricas de rendimiento')
    fig = px.bar(metricas, x='Métrica', y='Valor', title='Métricas del modelo RBF SVR', color='Métrica')
    st.plotly_chart(fig)

    # Recomendación específica
    st.subheader('Recomendación')
    if rbf_prediction > actual_values[0]:
        st.write("Recomendación: Comprar - Se espera que el precio suba mañana.")
    elif rbf_prediction < actual_values[0]:
        st.write("Recomendación: Vender - Se espera que el precio baje mañana.")
    else:
        st.write("Recomendación: Mantener - Se espera que el precio se mantenga estable mañana.")

