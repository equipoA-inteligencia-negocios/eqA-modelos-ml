import yfinance as yf
import ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score

# Configuración de estilo de Seaborn para gráficos
sns.set(style="whitegrid")

def show_corr_matrix(df, ticker):
    corr_matrix = df.corr()
    plt.figure(figsize=(16,14))
    sns.heatmap(corr_matrix[[f'Next_Close_{ticker}']], annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Matriz de correlación')
    st.write(f'#### Matriz de correlación de la variable objetivo Next_Close_{ticker} con las demás variables')
    st.pyplot(plt)

def calculate_indicators_mining_company(df, mining_company):
  # Hallamos precio de cierre del día previo
  df[f'Prev Close_{mining_company}'] = df[f'Close_{mining_company}'].shift(1)

  # Precio máximo del día previo
  df[f'Prev High_{mining_company}'] = df[f'High_{mining_company}'].shift(1)

  # Precio mínimo del día previo
  df[f'Prev Low_{mining_company}'] = df[f'Low_{mining_company}'].shift(1)

  # Precio de apertura del día anterior
  df[f'Prev Open_{mining_company}'] = df[f'Open_{mining_company}'].shift(1)

  # Indicador de fuerza relativa (RSI)
  df[f'RSI_{mining_company}'] = ta.momentum.RSIIndicator(df[f'Close_{mining_company}']).rsi()

  # Momentum (10 periodos)
  df[f'Momentum_{mining_company}'] = df[f'Close_{mining_company}'] - df[f'Close_{mining_company}'].shift(10)

  # Índices Estocásticos
  stoch = ta.momentum.StochasticOscillator(df[f'High_{mining_company}'], df[f'Low_{mining_company}'], df[f'Close_{mining_company}'])
  df[f'Stoch_K_{mining_company}'] = stoch.stoch()
  df[f'Stoch_D_{mining_company}'] = stoch.stoch_signal()

  # Promedios móviles
  df[f'SMA_50_{mining_company}'] = ta.trend.SMAIndicator(df[f'Close_{mining_company}'], window=50).sma_indicator()
  df[f'EMA_50_{mining_company}'] = ta.trend.EMAIndicator(df[f'Close_{mining_company}'], window=50).ema_indicator()

  # William %R
  df[f'WilliamsR_{mining_company}'] = ta.momentum.WilliamsRIndicator(df[f'High_{mining_company}'], df[f'Low_{mining_company}'], df[f'Close_{mining_company}']).williams_r()

  # Balance de volúmenes (OBV)
  df[f'OBV_{mining_company}'] = ta.volume.OnBalanceVolumeIndicator(df[f'Close_{mining_company}'], df[f'Volume_{mining_company}']).on_balance_volume()

  # Banda media y alta de Bollinger
  bollinger = ta.volatility.BollingerBands(df[f'Close_{mining_company}'], window=20)
  df[f'BB_Middle_{mining_company}'] = bollinger.bollinger_mavg()
  df[f'BB_Upper_{mining_company}'] = bollinger.bollinger_hband()

  # Precio medio (de la acción en el día)
  df[f'Avg Price_{mining_company}'] = (df[f'Open_{mining_company}'] + df[f'Close_{mining_company}']) / 2

  # Monto (Close * Volume)
  df[f'Amount_{mining_company}'] = df[f'Close_{mining_company}'] * df[f'Volume_{mining_company}']

  # BIAS (desviación del precio actual de su promedio)
  df[f'BIAS_{mining_company}'] = (df[f'Close_{mining_company}'] - df[f'SMA_50_{mining_company}']) / df[f'SMA_50_{mining_company}']

  # Indicador de cambio del volumen del precio (PVC)
  df[f'PVC_{mining_company}'] = df[f'Volume_{mining_company}'].pct_change()

  # Relación acumulativa (AR)
  df[f'AR_{mining_company}'] = df[f'High_{mining_company}'].rolling(window=14).mean() / df[f'Low_{mining_company}'].rolling(window=14).mean()

  # Movimiento promedio de Convergencia/Divergencia (MACD)
  macd = ta.trend.MACD(df[f'Close_{mining_company}'])
  df[f'MACD_{mining_company}'] = macd.macd()
  df[f'MACD_Signal_{mining_company}'] = macd.macd_signal()
  df[f'MACD_Hist_{mining_company}'] = macd.macd_diff()

  # Tasa de cambio (ROC)
  df[f'ROC_{mining_company}'] = ta.momentum.ROCIndicator(df[f'Close_{mining_company}']).roc()

  # Línea psicológica (PSY)
  df[f'PSY_{mining_company}'] = df[f'Close_{mining_company}'].rolling(window=12).apply(lambda x: np.sum(x > x.shift(1)) / len(x))

  # Diferencia (DIF)
  df[f'DIF_{mining_company}'] = df[f'Close_{mining_company}'].diff()

  return df

def plot_moving_average(data, ticker, window_size=50):
    # Calcular la media móvil
    data_sma = data.copy()
    data_sma[f'SMA_{window_size}'] = data_sma[f'Close_{ticker}'].rolling(window=window_size).mean()
    
    # Plotear los precios reales y la media móvil
    plt.figure(figsize=(14, 7))
    plt.plot(data_sma.index, data_sma[f'Close_{ticker}'], label='Precio Real')
    plt.plot(data_sma.index, data_sma[f'SMA_{window_size}'], label=f'Media Móvil {window_size}')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title(f'Precio Real vs Media Móvil de {ticker}')
    plt.legend()
    st.pyplot(plt)
    
def predict_next_day_price(model, data, seq_length, scaler, ticker):
    last_sequence = data.iloc[-seq_length:].values.reshape(1, seq_length, -1)
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(
        np.concatenate((np.zeros((1, data.shape[1] - 1)), next_day_scaled), axis=1)
    )[:, -1]
    return next_day_price[0]

def create_sequences(data, seq_length, ticker):
    sequences = []
    labels = []
    indices = []
    for i in range(len(data) - seq_length):
        # Acceso posicional con .iloc
        sequences.append(data.iloc[i:i + seq_length].values)
        # Añadir el precio del día siguiente como etiqueta
        labels.append(data.iloc[i + seq_length][f'Next_Close_{ticker}'])
        # Formatear el índice como "Fecha de Inicio - Fecha de Fin"
        start_date = data.index[i].strftime('%Y-%m-%d')
        end_date = data.index[i + seq_length - 1].strftime('%Y-%m-%d')
        indices.append(f"{start_date} - {end_date}")

    # Crear DataFrame de secuencias
    sequences_df = pd.DataFrame({
        'Sequence': sequences,
        f'Next_Close_{ticker}': labels
    }, index=indices)

    return sequences_df

# Función para descargar y procesar datos de Yahoo Finance
def descargar_datos(ticker, fecha_inicio, fecha_fin):
    """## Minera"""
    
    data = yf.download(ticker, start=fecha_inicio, end=fecha_fin)
    data.columns += f"_{ticker}"
    
    # additional_data_list = get_additional_data(fecha_inicio, fecha_fin)
    
    # additional_data = pd.concat(additional_data_list, axis=1)
    
    # [additional_data] = get_additional_data(fecha_inicio, fecha_fin)
    return data

# Función para preprocesar datos
def preprocesar_datos(data, ticker, fecha_inicio, fecha_fin):
    IGBVL_data = yf.download('^SPBLPGPT', start = fecha_inicio, end = fecha_fin)

    # Removiendo columna 'Volume' y 'Adj Close'
    IGBVL_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

    IGBVL_data.columns += "_IGBVL"
    
    """### Datos DJI"""

    DJI_data = yf.download('^DJI', start = fecha_inicio, end = fecha_fin)

    # Removiendo columna 'Volume' y 'Adj Close'
    DJI_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

    DJI_data.columns += "_DJI"

    """### Datos NASDAQ"""

    NASDAQ_data = yf.download('^IXIC', start = fecha_inicio, end = fecha_fin)

    # Removiendo columna 'Volume' y 'Adj Close'
    NASDAQ_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

    NASDAQ_data.columns += "_NASDAQ"

    """### Datos del dólar"""

    PEN_X_data = yf.download('PEN=X', start = fecha_inicio, end = fecha_fin)

    # Removiendo columna 'Volume' y 'Adj Close'
    PEN_X_data.drop( ['Adj Close', 'Volume'] , axis=1, inplace=True)

    PEN_X_data.columns += "_PEN_X"

    """### Datos del oro"""

    GLD_data = yf.download('GLD', start = fecha_inicio, end = fecha_fin)

    # Removiendo columna 'Volume' y 'Close'
    GLD_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

    GLD_data.columns += "_GLD"

    """### Datos de la plata"""

    SIF_data = yf.download('SI=F', start = fecha_inicio, end = fecha_fin)

    # Removiendo columna 'Volume' y 'Adj Close'
    SIF_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

    SIF_data.columns += "_SIF"

    """### Datos del cobre"""

    HGF_data = yf.download('HG=F', start = fecha_inicio, end = fecha_fin)

    # Removiendo columna 'Volume' y 'Close'
    HGF_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

    HGF_data.columns += "_HGF"

    """### Datos del zinc"""

    T09_ZINC = yf.download('ZINC.L', start=fecha_inicio, end=fecha_fin)

    # Removiendo columna 'Volume' y 'Close'
    T09_ZINC.drop(['Volume', 'Adj Close'], axis=1, inplace=True)

    T09_ZINC.columns += "_ZINC"
    
    # Calculamos los indicadores de la empresa Buenaventura
    data = calculate_indicators_mining_company(data, ticker)

    data[f'Next_Close_{ticker}'] = data[f'Close_{ticker}'].shift(-1)
    
    data_v2 = data[[f'Open_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Close_{ticker}', f'Adj Close_{ticker}', f'Prev Close_{ticker}', f'Prev High_{ticker}', f'Prev Low_{ticker}', f'Prev Open_{ticker}', f'SMA_50_{ticker}', f'EMA_50_{ticker}', f'BB_Middle_{ticker}', f'BB_Upper_{ticker}', f'Avg Price_{ticker}', f'Next_Close_{ticker}']]
    
    st.dataframe(data_v2)
    
    IGBVL_data = IGBVL_data[['Close_IGBVL']]
    DJI_data = DJI_data[['Close_DJI']]
    NASDAQ_data = NASDAQ_data[['Close_NASDAQ']]
    PEN_X_data = PEN_X_data[['Close_PEN_X']]
    GLD_data = GLD_data[['Close_GLD']]
    SIF_data = SIF_data[['Close_SIF']]
    HGF_data = HGF_data[['Close_HGF']]
    T09_ZINC = T09_ZINC[['Close_ZINC']]
    
    
    if IGBVL_data.empty:
        data_v3 = pd.merge(data_v2, DJI_data, on='Date')
        data_v3 = pd.merge(data_v3, NASDAQ_data, on='Date')
        data_v3 = pd.merge(data_v3, PEN_X_data, on='Date')
        data_v3 = pd.merge(data_v3, GLD_data, on='Date')
        data_v3 = pd.merge(data_v3, SIF_data, on='Date')
        data_v3 = pd.merge(data_v3, HGF_data, on='Date')
        data_v3 = pd.merge(data_v3, T09_ZINC, on='Date')
    else:
        data_v3 = pd.merge(data_v2, IGBVL_data, on='Date')
        data_v3 = pd.merge(data_v3, DJI_data, on='Date')
        data_v3 = pd.merge(data_v3, NASDAQ_data, on='Date')
        data_v3 = pd.merge(data_v3, PEN_X_data, on='Date')
        data_v3 = pd.merge(data_v3, GLD_data, on='Date')
        data_v3 = pd.merge(data_v3, SIF_data, on='Date')
        data_v3 = pd.merge(data_v3, HGF_data, on='Date')
        data_v3 = pd.merge(data_v3, T09_ZINC, on='Date')
    
    # if (not data_v3.empty):
    #     show_corr_matrix(data_v3, ticker)
    # else:
    #     show_corr_matrix(data_v2, ticker)
    
    show_corr_matrix(data_v3, ticker)
    
    if (ticker == 'FSM'):
        data_v4 = pd.merge(data_v2, SIF_data, on='Date')
    else:
        data_v4 = data_v2
    
    st.write("#### Dataframe con las variables adicionales")
    st.dataframe(data_v4)

    # Escalar los datos
    # scaler = MinMaxScaler(feature_range=(0, 1))

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    columns_X = data_v4.columns.drop(f'Next_Close_{ticker}')
    
    # Normalizamos las variables independientes
    data_v4.loc[:, columns_X] = scaler_X.fit_transform(data_v4[columns_X])
    
    # Normalizamos la variable dependiente
    data_v4.loc[:, f'Next_Close_{ticker}'] = scaler_y.fit_transform(data_v4[f'Next_Close_{ticker}'].values.reshape(-1,1)).flatten()
    
    # columns = data_v4.columns

    # data_v4[columns] = scaler.fit_transform(data_v4[columns])

    # Por último, el tratamiento de valores nulos empleado en la tesis es de eliminarlos
    # data_v4.isnull().sum()

    data_v4.dropna(inplace=True)
    
    st.write("#### Dataframe normalizado")
    st.dataframe(data_v4)
    
    return data_v4, scaler_y

# Función para entrenar el modelo
def entrenar_modelo(data, ticker):
    seq_length = 60
    sequences_df = create_sequences(data, seq_length, ticker)
    
    train_df, test_df = train_test_split(sequences_df, test_size=0.2, random_state=42)

    # Separar las secuencias y etiquetas
    X_train = np.array(train_df['Sequence'].tolist())
    y_train = np.array(train_df[f'Next_Close_{ticker}'].tolist())
    X_test = np.array(test_df['Sequence'].tolist())
    y_test = np.array(test_df[f'Next_Close_{ticker}'].tolist())

    # Asegurar que mantenemos los índices
    X_train_indices = train_df.index
    X_test_indices = test_df.index
    y_train_indices = train_df.index
    y_test_indices = test_df.index
    
    # Construimos el modelo LSTM
    model = Sequential()

    # Añadimos una capa LSTM con 50 unidades y función de activación ReLU
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, X_train.shape[2])))
    # LSTM(50): Añade una capa LSTM con 50 neuronas o unidades LSTM.
    # activation='relu': Utiliza la función de activación ReLU (Rectified Linear Unit), que es una función no lineal común en redes neuronales que ayuda a manejar problemas de gradientes desvanecidos.
    # input_shape=(timesteps, n_features): Define la forma de la entrada. Aquí, 'timesteps' es el número de pasos de tiempo y 'n_features' es el número de características en cada paso de tiempo.

    # Añadir una capa densa con una unidad de salida
    model.add(Dense(1))
    # Añade una capa completamente conectada con 1 neurona. Esta es la capa de salida del modelo, que proporciona la predicción final.
    # No se especifica una función de activación en esta capa, por lo que se utiliza la activación lineal por defecto.
    # Esto es adecuado para problemas de regresión donde la salida es un valor continuo, como en este caso que queremos predecir un precio.

    # Compilación del modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    # optimizer=Adam(learning_rate=0.001): Utiliza el optimizador Adam con una tasa de aprendizaje de 0.001
    # loss='mean_squared_error': Utiliza la función de pérdida de error cuadrático medio (MSE)

    # Entrenamos el modelo con 50 épocas y un tamaño de lote de 32
    history = model.fit(
        X_train,       # Datos de entrada para el entrenamiento
        y_train,                # Etiquetas correspondientes
        epochs=50,              # Número de veces que el modelo verá el conjunto completo de datos de entrenamiento
        batch_size=32,          # Número de muestras que se procesan antes de actualizar el modelo
        validation_split=0.2,   # Porcentaje del conjunto de entrenamiento utilizado para la validación
        verbose=2               # Modo de verbosidad; 2 para una salida detallada por época
    )
    
    print(model)
    
    return model, X_test, y_test, y_test_indices

# def evaluate_model(ticker, model, X_test, y_test, y_test_indices):
#     # Evaluamos la pérdida del modelo (MSE) en el conjunto de prueba
#     loss = model.evaluate(X_test, y_test, verbose=0)
#     print(f'Pérdida en el conjunto de prueba (MSE): {loss} ({loss:.8f})')

#     # Calculamos el RMSE en el conjunto de prueba
#     rmse = np.sqrt(loss)
#     print(f'RMSE en el conjunto de prueba: {rmse}')

#     # Hacemos predicciones con el modelo entrenado
#     y_pred = model.predict(X_test)

#     # Mostramos las predicciones y los valores reales
#     # Creamos un dataframe para ello
#     results = pd.DataFrame({'Real': y_test.flatten(), 'Predicted': y_pred.flatten()})
#     results.head()
    
#     # Convertimos el conjunto de prueba a una dimensión
#     y_test_flatten = y_test.flatten()
#     # Convertimos las predicciones a una dimensión
#     y_pred_flatten = y_pred.flatten()

#     # MAPE (Mean Absolute Percentage Error)
#     mape = mean_absolute_percentage_error(y_test_flatten, y_pred_flatten)

#     # RMSE (Root Mean Square Error)
#     rmse = np.sqrt(mean_squared_error(y_test_flatten, y_pred_flatten))

#     # Mostrar las métricas
#     print(f'MAPE: {mape:.4f}')
#     print(f'RMSE: {rmse:.4f}')
    
#     results = pd.DataFrame({
#         'Fecha': y_test_indices,
#         'Valor Real': y_test_flatten,
#         'Predicción Modelo LSTM': y_pred_flatten
#     })

#     # Ordenar el DataFrame por fecha
#     results.sort_values(by='Fecha', inplace=True)

#     # Visualización de resultados
#     plt.figure(figsize=(14, 7))
#     plt.plot(results['Fecha'], results['Valor Real'], label='Valor Real')
#     plt.plot(results['Fecha'], results['Predicción Modelo LSTM'], label='Predicción Modelo LSTM')
#     plt.xlabel('Fecha')
#     plt.ylabel('Precio de Cierre')
#     plt.title(f'Comparación del Valor Real vs Predicción del Modelo LSTM para {ticker}')
#     plt.legend()
#     st.pyplot(plt)

def evaluate_model(ticker, model, X_test, y_test, y_test_indices):
    # Evaluamos la pérdida del modelo (MSE) en el conjunto de prueba
    loss = model.evaluate(X_test, y_test, verbose=0)

    # Calculamos el RMSE en el conjunto de prueba
    rmse = np.sqrt(loss)

    # st.subheader("Predicciones en el conjunto de prueba")
    
    # Hacemos predicciones con el modelo entrenado
    y_pred = model.predict(X_test)

    # Mostramos las predicciones y los valores reales
    # Creamos un dataframe para ello
    results = pd.DataFrame({'Real': y_test.flatten(), 'Predicted': y_pred.flatten()})
    results.head()

    # Convertimos el conjunto de prueba a una dimensión
    y_test_flatten = y_test.flatten()
    # Convertimos las predicciones a una dimensión
    y_pred_flatten = y_pred.flatten()

    # MAPE (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(y_test_flatten, y_pred_flatten)

    # RMSE (Root Mean Square Error)
    # rmse = np.sqrt(mean_squared_error(y_test_flatten, y_pred_flatten))

    # R^2 (coeficiente de determinación)
    r2 = r2_score(y_test_flatten, y_pred_flatten)
    
    # st.subheader("Predicciones en el conjunto de prueba")

    # Mostrar las métricas con Streamlit
    st.write(f"Pérdida en el conjunto de prueba (MSE): {loss:.8f}")
    st.write(f"RMSE en el conjunto de prueba: {rmse:.4f}")
    st.write(f"MAPE: {mape:.4f}")
    st.write(f"Coeficiente de determinación (R^2): {r2:.4f}")

    results = pd.DataFrame({
        'Fecha': y_test_indices,
        'Valor Real': y_test_flatten,
        'Predicción Modelo LSTM': y_pred_flatten
    })

    # Ordenar el DataFrame por fecha
    results.sort_values(by='Fecha', inplace=True)
    
    st.subheader("Predicciones en el conjunto de prueba")

    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(results['Fecha'], results['Valor Real'], label='Valor Real')
    plt.plot(results['Fecha'], results['Predicción Modelo LSTM'], label='Predicción Modelo LSTM')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title(f'Comparación del Valor Real vs Predicción del Modelo LSTM para {ticker}')
    plt.legend()
    st.pyplot(plt)
    
def create_sequences_full(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length].values)
    return np.array(sequences)

def plot_real_prices_full(seq_length, data, model, scaler_y, ticker):
    # Configuración del modelo LSTM
    X_full = create_sequences_full(data, seq_length)

    # Realizar predicciones en todo el conjunto de datos
    y_pred_full = model.predict(X_full)
    
    

    # Desescalar las predicciones y los valores reales
    # full_indices = data.index[seq_length:]
    y_pred_full_descaled = scaler_y.inverse_transform(y_pred_full)
    y_real_full_descaled = scaler_y.inverse_transform(data.iloc[seq_length:][f'Next_Close_{ticker}'].values.reshape(-1, 1))

    # Crear el DataFrame con los valores desescalados
    results_full = pd.DataFrame({
        'Fecha': data.index[seq_length:seq_length + len(y_pred_full_descaled)],
        'Valor Real': y_real_full_descaled.flatten(),
        'Predicción Modelo LSTM': y_pred_full_descaled.flatten()
    })

    # Ordenar el DataFrame por fecha
    results_full.sort_values(by='Fecha', inplace=True)

    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(results_full['Fecha'], results_full['Valor Real'], label='Valor Real')
    plt.plot(results_full['Fecha'], results_full['Predicción Modelo LSTM'], label='Predicción Modelo LSTM')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title(f'Comparación del Valor Real vs Predicción del Modelo LSTM para {ticker}')
    plt.legend()
    st.pyplot(plt)

    return results_full

def app():
    st.title("Modelo de Predicción de Precios de Acciones")
    
    st.write("""
    ### Bienvenido a la aplicación de predicción de precios de acciones. 
    Esta aplicación utiliza un modelo LSTM para predecir los precios de cierre de acciones y determinar la tendencia del mercado.
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
        st.write(f"""
        Se crea una columna 'Next_Close_{ticker}' que contiene el precio de cierre del día siguiente.
        """)
        
        # datos = preprocesar_datos(datos, ticker, fecha_inicio, fecha_fin)
        datos, scaler = preprocesar_datos(datos, ticker, fecha_inicio, fecha_fin)

        # Ploteo de la media móvil
        st.write("#### Media Móvil de los Precios Reales")
        plot_moving_average(datos, ticker)

        # Entrenar modelo
        st.subheader("Entrenamiento del Modelo")
        st.write("""
        El modelo se entrena utilizando un LSTM. 
        Este modelo de aprendizaje automático es eficaz para la predicción de series temporales debido a su capacidad para manejar secuencias de datos.
        """)
        st.write("Entrenando el modelo...")
        model, X_test, y_test, y_test_indices = entrenar_modelo(datos, ticker)
        st.write("Modelo entrenado!")
        
        # Realizar predicciones
        st.subheader("Evaluación del Modelo")
        st.write("""
        Después de entrenar el modelo, se utilizan los datos de prueba para realizar predicciones sobre la tendencia de los precios de cierre para los días siguientes.
        """)
        st.subheader(f"Evaluación del modelo para {ticker}")
        evaluate_model(ticker, model, X_test, y_test, y_test_indices)
        
        df_predicciones = plot_real_prices_full(60, datos, model, scaler, ticker)
        
        st.subheader("Dataframe de predicciones en el conjunto de datos completo")
        st.dataframe(df_predicciones)

        # Predicción del precio del día siguiente
        next_day_price = predict_next_day_price(model, datos, 60, scaler, ticker)
        st.subheader("Predicción del Precio del Día Siguiente")
        st.write(f"El precio de {ticker} se pronóstica según el modelo LSTM para el siguiente día como: {next_day_price:.2f} por acción.")