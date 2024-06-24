import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Agregamos la importación de seaborn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def collect_data(start_date, end_date):
    # Descargar datos de Yahoo Finance para BVN
    df_bvn = yf.download('BVN', start=start_date, end=end_date)

    return df_bvn

def preprocess_data(df):
    # Cálculo de indicadores técnicos
    df = calculate_indicators(df)

    # Selección de variables relevantes
    variables_relevantes = ['Open', 'High', 'Low', 'Close', 'Prev Close', 'Prev High', 'Prev Low', 'Prev Open', 'SMA_50', 'EMA_50', 'BB_Middle', 'BB_Upper', 'Avg Price', 'Target_Close']
    df_v2 = df[variables_relevantes]

    return df_v2

def calculate_indicators(df):
    # Hallamos precio de cierre del día previo
    df['Prev Close'] = df['Close'].shift(1)

    # Precio máximo del día previo
    df['Prev High'] = df['High'].shift(1)

    # Precio mínimo del día previo
    df['Prev Low'] = df['Low'].shift(1)

    # Precio de apertura del día anterior
    df['Prev Open'] = df['Open'].shift(1)

    # Indicador de fuerza relativa (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    # Momentum (10 periodos)
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # Índices Estocásticos
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # Promedios móviles
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()

    # Banda media y alta de Bollinger
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Upper'] = bollinger.bollinger_hband()

    # Precio medio (de la acción en el día)
    df['Avg Price'] = (df['Open'] + df['Close']) / 2

    # Monto (Close * Volume)
    df['Amount'] = df['Close'] * df['Volume']

    # BIAS (desviación del precio actual de su promedio)
    df['BIAS'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    # Indicador de cambio del volumen del precio (PVC)
    df['PVC'] = df['Volume'].pct_change()

    # Relación acumulativa (AR)
    df['AR'] = df['High'].rolling(window=14).mean() / df['Low'].rolling(window=14).mean()

    # Movimiento promedio de Convergencia/Divergencia (MACD)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # Tasa de cambio (ROC)
    df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()

    # Línea psicológica (PSY)
    df['PSY'] = df['Close'].rolling(window=12).apply(lambda x: np.sum(x > x.shift(1)) / len(x))

    # Diferencia (DIF)
    df['DIF'] = df['Close'].diff()

    # Crear la columna de precio de cierre del día siguiente
    df['Target_Close'] = df['Close'].shift(-1)

    # Eliminar filas con valores NaN
    df.dropna(inplace=True)

    return df

def train_and_evaluate_knn(X_train, y_train, X_test, y_test, k=5):
    model_knn = KNeighborsRegressor(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    
    y_pred_train = model_knn.predict(X_train)
    y_pred_test = model_knn.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    return model_knn, mse_train, r2_train, mse_test, r2_test

def app():
    st.title('Minería y Finanzas')

    st.sidebar.header('Recolección de Datos')
    start_date = st.sidebar.date_input('Fecha de inicio', pd.to_datetime('2018-01-01'))
    end_date = st.sidebar.date_input('Fecha de fin', pd.to_datetime('2022-12-31'))

    st.sidebar.write('---')

    st.sidebar.header('Modelado y Entrenamiento')
    k_value = st.sidebar.slider('Número de Vecinos (K)', 1, 20, 5)

    st.sidebar.write('---')

    if st.sidebar.button('Entrenar y Evaluar'):
        st.write('Entrenando y evaluando el modelo...')

        # Recolectar datos
        df_bvn = collect_data(start_date, end_date)

        # Preprocesar datos
        df_bvn_v2 = preprocess_data(df_bvn)

        # Dividir datos para entrenamiento y prueba
        X = df_bvn_v2.drop(columns=['Target_Close'])
        y = df_bvn_v2['Target_Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalización de datos
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenamiento y evaluación del modelo KNN
        model_knn, mse_train, r2_train, mse_test, r2_test = train_and_evaluate_knn(X_train_scaled, y_train, X_test_scaled, y_test, k=k_value)

        # Mostrar resultados
        st.write('### Resultados de la Evaluación:')
        st.write(f'- MSE Train: {mse_train:.4f}, R2 Train: {r2_train:.4f}')
        st.write(f'- MSE Test: {mse_test:.4f}, R2 Test: {r2_test:.4f}')

       # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from pandas_datareader import data as pdr
import yfinance as yfin
import plotly.express as px

def collect_data(start_date, end_date):
    df_bvn = yfin.download('BVN', start=start_date, end=end_date)
    return df_bvn

def preprocess_data(df):
    return df

def train_and_evaluate_knn(X_train, y_train, X_test, y_test, k=5):
    model_knn = KNeighborsRegressor(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    
    y_pred_train = model_knn.predict(X_train)
    y_pred_test = model_knn.predict(X_test)
    
    mse_train = metrics.mean_squared_error(y_train, y_pred_train)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    
    mse_test = metrics.mean_squared_error(y_test, y_pred_test)
    r2_test = metrics.r2_score(y_test, y_pred_test)
    
    return model_knn, mse_train, r2_train, mse_test, r2_test

def app():
    st.title('Modelo KNN para Predicción de Precios de Mineras')
    st.subheader('Seleccione los parámetros para la predicción')

    plt.style.use('fivethirtyeight')

    yfin.pdr_override()

    st.subheader("Obtener datos de Yahoo Finance")
    start = st.date_input('Inicio', value=pd.to_datetime('2014-01-01'))
    end = st.date_input('Fin', value=pd.to_datetime('2014-04-28'))
    stock_option = st.selectbox('Seleccione la cotización bursátil', ['BVN', 'FSM'])

    df = pdr.get_data_yahoo(stock_option, start, end)
    st.write(df)
    st.subheader("Detalles de los datos")
    st.write(df.describe())
    st.write("Número de filas y columnas:")
    st.write(df.shape)

    # Obtener los datos hasta el penúltimo día
    df_train = df.head(len(df) - 1)
    df_test = df.tail(1)
    
    X_train = df_train.drop(columns=['Adj Close'])
    y_train = df_train['Adj Close']
    X_test = df_test.drop(columns=['Adj Close'])
    y_test = df_test['Adj Close']

    # Normalización de datos (no se realiza en este caso)

    # Entrenamiento y evaluación del modelo KNN
    model_knn, mse_train, r2_train, mse_test, r2_test = train_and_evaluate_knn(X_train, y_train, X_test, y_test)

    # Predicción para el día siguiente
    prediction_day = X_test.index[-1] + pd.DateOffset(days=1)
    knn_prediction = model_knn.predict([X_test.iloc[-1]])

    st.subheader('Predicción para el día siguiente')
    st.write(f'Predicción KNN: {knn_prediction}')

    # Evaluación del modelo
    actual_values = [y_test.values[0]]
    predicted_values = [knn_prediction]

    MAE = metrics.mean_absolute_error(actual_values, predicted_values)
    MSE = metrics.mean_squared_error(actual_values, predicted_values)
    RMSE = np.sqrt(metrics.mean_squared_error(actual_values, predicted_values))

    metricas = pd.DataFrame({
        'Métrica': ['Error Absoluto Medio', 'Error Cuadrático Medio', 'Raíz del Error Cuadrático Medio'],
        'Valor': [MAE, MSE, RMSE]
    })

    # Gráfico de barras para valores reales y valor del día siguiente
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred_next_day = model_knn.predict(X_test_scaled[-1].reshape(1, -1))

    df_resultados = pd.DataFrame({
        'Tipo': ['Real', 'Real del Día Siguiente'],
        'Valor': [y_test.values[-1], y_pred_next_day[0]]
    })

    fig = px.bar(df_resultados, x='Tipo', y='Valor', title='Comparación de Valores Reales y Valor del Día Siguiente')
    st.plotly_chart(fig)

    # Calcular la media móvil
    rolling_mean = df['Adj Close'].rolling(window=20).mean()

    # Dividir el eje y por mes
    monthly_dates = df.resample('M').mean().index
    monthly_ticks = df.resample('M').mean()['Adj Close']

    # Obtener los valores reales
    real_values = df['Adj Close']

    plt.figure(figsize=(10, 6))

    # Graficar los valores reales
    plt.plot(df.index, real_values, marker='o', linestyle='-', color='blue', label='Valor Real')

    # Graficar la media móvil
    plt.plot(df.index, rolling_mean, color='red', linestyle='-', label='Media Móvil (20 días)')

    # Configurar el eje y para que esté dividido por mes
    plt.yticks(monthly_ticks, [date.strftime('%Y-%m') for date in monthly_dates])

    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.title('Comparación de Valores Reales y Media Móvil')
    plt.legend()
    st.pyplot(plt)

    # Recomendación específica
    st.subheader('Recomendación')
    if knn_prediction > actual_values[0]:
        st.write("Recomendación: Comprar - Se espera que el precio suba mañana.")
    elif knn_prediction < actual_values[0]:
        st.write("Recomendación: Vender - Se espera que el precio baje mañana.")
    else:
        st.write("Recomendación: Mantener - Se espera que el precio se mantenga estable mañana.")


