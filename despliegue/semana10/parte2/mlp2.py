import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from ta.momentum import RSIIndicator
from ta.trend import MACD
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def download_data(ticker, start_date, end_date):
    df_final = yf.download(ticker, start=start_date, end=end_date)
    df_final.drop('Volume', axis=1, inplace=True)
    return df_final

def calculate_indicators(df, ticker):
    df[f'Prev Close_{ticker}'] = df['Close'].shift(1)
    df[f'Prev High_{ticker}'] = df['High'].shift(1)
    df[f'Prev Low_{ticker}'] = df['Low'].shift(1)
    df[f'Prev Open_{ticker}'] = df['Open'].shift(1)
    df[f'SMA_50_{ticker}'] = df['Close'].rolling(window=50).mean()
    df[f'EMA_50_{ticker}'] = df['Close'].ewm(span=50, adjust=False).mean()
    df[f'BB_Middle_{ticker}'] = df[f'SMA_50_{ticker}']
    df[f'BB_Upper_{ticker}'] = df[f'BB_Middle_{ticker}'] + 2 * df['Close'].rolling(window=20).std()
    df[f'Avg Price_{ticker}'] = (df['High'] + df['Low']) / 2
    df[f'Next_Close_{ticker}'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length, ticker):
    sequences = []
    labels = []
    indices = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length].values)
        labels.append(data.iloc[i + seq_length][f'Next_Close_{ticker}'])
        start_date = data.index[i].strftime('%Y-%m-%d')
        end_date = data.index[i + seq_length - 1].strftime('%Y-%m-%d')
        indices.append(f"{start_date} - {end_date}")

    sequences_df = pd.DataFrame({
        'Sequence': sequences,
        f'Next_Close_{ticker}': labels
    }, index=indices)

    return sequences_df

def plot_histogram(df, column_name, title):
    fig, ax = plt.subplots()
    sns.histplot(df[column_name], kde=True, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_corr_matrix(df, ticker):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix[[f'Next_Close_{ticker}']], annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
    ax.set_title(f'Matriz de correlación para {ticker}')
    st.pyplot(fig)

def plot_training_loss(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    ax.plot(history.history['val_loss'], label='Pérdida de Validación')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Pérdida')
    ax.set_title('Evolución de la Pérdida durante el Entrenamiento')
    ax.legend()
    st.pyplot(fig)

def plot_predictions_vs_real(y_test, y_pred, ticker):
    fig, ax = plt.subplots()
    ax.plot(y_test, label='Valor Real')
    ax.plot(y_pred, label='Predicción')
    ax.set_xlabel('Observaciones')
    ax.set_ylabel('Precio de Cierre')
    ax.set_title(f'Comparación del Valor Real vs Predicción del Modelo para {ticker} en el Conjunto de Prueba')
    ax.legend()
    st.pyplot(fig)

def plot_scatter(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Predicho')
    ax.set_title('Gráfico de Dispersión de Valores Reales vs Predichos')
    st.pyplot(fig)

def plot_real_prices_full(seq_length, data, model, scaler_y, ticker):
    # Crear secuencias y etiquetas a partir de los datos completos
    sequences_df = create_sequences(data, seq_length, ticker)
    
    # Convertir la columna 'Sequence' a arreglos numpy
    X_full = np.array(sequences_df['Sequence'].tolist())
    X_full_flat = np.array([x.flatten() for x in X_full])
    
    # Realizar predicciones
    y_pred_full = model.predict(X_full_flat)
    
    # Desescalar las predicciones
    y_pred_full_descaled = scaler_y.inverse_transform(y_pred_full)
    
    # Desescalar los valores reales
    y_real_full_descaled = scaler_y.inverse_transform(data.iloc[seq_length:][f'Next_Close_{ticker}'].values.reshape(-1, 1))
    
    # Crear un DataFrame con los resultados
    result_df = pd.DataFrame({
        'Fecha': data.index[seq_length:],
        'Valor Real': y_real_full_descaled.flatten(),
        'Predicción': y_pred_full_descaled.flatten()
    })
    
    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result_df['Fecha'], result_df['Valor Real'], label='Valor Real')
    ax.plot(result_df['Fecha'], result_df['Predicción'], label='Predicción Modelo MLP')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de Cierre')
    ax.set_title(f'Comparación del Valor Real vs Predicción del Modelo MLP para {ticker}')
    ax.legend()
    st.pyplot(fig)
    
    return result_df

def predict_next_day_price(model, data, seq_length, scaler_y, ticker):
    last_sequence = data.iloc[-seq_length:].values.reshape(1, seq_length * data.shape[1])
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler_y.inverse_transform(next_day_scaled.reshape(-1, 1))
    return next_day_price[0][0]

def app():
    st.title('Predicción de Precios con MLP')
    st.header('Configuración')

    # Parámetros del usuario
    ticker = st.selectbox('Ticker', ['BVN', 'FSM', 'SCCO'])
    start_date = st.date_input('Fecha de Inicio', pd.to_datetime('2018-01-01'))
    end_date = st.date_input('Fecha de Fin', pd.to_datetime('2022-12-31'))
    seq_length = 60
    epochs = 50
    batch_size = 32

    # Descargar y preparar los datos
    df = download_data(ticker, start_date, end_date)
    df = calculate_indicators(df, ticker)
    
    st.write(f"Se tienen los siguientes datos para {ticker}:")
    st.dataframe(df.head())

    # Calcular RSI y MACD
    st.write(f'##Calculamos el RSI y MACD para {ticker}.')
    df[f'RSI_{ticker}'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df[f'MACD_{ticker}'] = macd.macd()
    df[f'MACD_signal_{ticker}'] = macd.macd_signal()
    df[f'MACD_diff_{ticker}'] = macd.macd_diff()

    plot_histogram(df, f'RSI_{ticker}', f'RSI_{ticker}')
    plot_histogram(df, f'MACD_{ticker}', f'MACD_{ticker}')
    plot_histogram(df, f'MACD_signal_{ticker}', f'MACD_signal_{ticker}')
    plot_histogram(df, f'MACD_diff_{ticker}', f'MACD_diff_{ticker}')

    st.write(f'##Datos con la variable objetivo Next_Close_{ticker}')
    st.dataframe(df)

    st.write(f'##Matriz de correlación de la variable objetivo Next_Close_{ticker}')
    plot_corr_matrix(df, ticker)

    st.write(f'##Dataframe con las variables seleccionadas para {ticker}')
    selected_columns = [
        f'Prev Close_{ticker}', f'Prev High_{ticker}', f'Prev Low_{ticker}', f'Prev Open_{ticker}', 
        f'SMA_50_{ticker}', f'EMA_50_{ticker}', f'BB_Middle_{ticker}', f'BB_Upper_{ticker}', 
        f'Avg Price_{ticker}', f'RSI_{ticker}', f'MACD_{ticker}', f'MACD_signal_{ticker}', 
        f'MACD_diff_{ticker}', f'Next_Close_{ticker}'
    ]
    df_final = df[selected_columns]
    df_final.dropna(inplace=True)
    st.dataframe(df_final)

    st.write('## Dataframe normalizado')
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    df_final.loc[:, selected_columns[:-1]] = scaler_X.fit_transform(df_final[selected_columns[:-1]])
    df_final[f'Next_Close_{ticker}'] = scaler_y.fit_transform(df_final[[f'Next_Close_{ticker}']])
    st.dataframe(df_final)

    st.write('## Crear secuencias para el entrenamiento y prueba')
    sequences_df = create_sequences(df_final, seq_length, ticker)
    # Convertir la columna 'Sequence' a una representación de cadena
    sequences_df['Sequence'] = sequences_df['Sequence'].apply(lambda x: x.tolist())
    st.dataframe(sequences_df.head())

    st.write('## Preparar datos para el modelo MLP')
    X = np.array(sequences_df['Sequence'].tolist())
    y = np.array(sequences_df[f'Next_Close_{ticker}'].tolist())
    X_flat = np.array([x.flatten() for x in X])
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    # Definir modelo MLP
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compilar modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Entrenar modelo
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=2
    )

    # Evaluar modelo
    loss = model.evaluate(X_test, y_test, verbose=0)
    rmse = np.sqrt(loss)
    mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
    
    st.write(f'## Evaluación del modelo')
    st.write(f'Pérdida en el conjunto de prueba (MSE): {loss:.8f}')
    st.write(f'RMSE en el conjunto de prueba: {rmse:.4f}')
    st.write(f'MAPE: {mape:.4f}')

    st.write('## Gráfico de pérdida durante el entrenamiento')
    plot_training_loss(history)

    st.write('## Comparación de valores reales vs predichos')
    y_pred = model.predict(X_test)
    plot_predictions_vs_real(y_test, y_pred, ticker)

    st.write('## Gráfico de dispersión de valores reales vs predichos')
    plot_scatter(y_test, y_pred)

    st.write(f'## Comparación del Valor Real vs Predicción del Modelo MLP para {ticker}')
    result_df = plot_real_prices_full(seq_length, df_final, model, scaler_y, ticker)
    st.dataframe(result_df)

    st.write('## Predicción del precio del siguiente día')
    next_day_price = predict_next_day_price(model, df_final, seq_length, scaler_y, ticker)
    st.write(f'El precio predicho del siguiente día para {ticker} es: {next_day_price:.2f}')

if __name__ == '__main__':
    app()
# Importamos las librerías necesarias
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from ta.momentum import RSIIndicator
from ta.trend import MACD
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def download_data(ticker, start_date, end_date):
    df_final = yf.download(ticker, start=start_date, end=end_date)
    df_final.drop('Volume', axis=1, inplace=True)
    return df_final

def calculate_indicators(df, ticker):
    df[f'Prev Close_{ticker}'] = df['Close'].shift(1)
    df[f'Prev High_{ticker}'] = df['High'].shift(1)
    df[f'Prev Low_{ticker}'] = df['Low'].shift(1)
    df[f'Prev Open_{ticker}'] = df['Open'].shift(1)
    df[f'SMA_50_{ticker}'] = df['Close'].rolling(window=50).mean()
    df[f'EMA_50_{ticker}'] = df['Close'].ewm(span=50, adjust=False).mean()
    df[f'BB_Middle_{ticker}'] = df[f'SMA_50_{ticker}']
    df[f'BB_Upper_{ticker}'] = df[f'BB_Middle_{ticker}'] + 2 * df['Close'].rolling(window=20).std()
    df[f'Avg Price_{ticker}'] = (df['High'] + df['Low']) / 2
    df[f'Next_Close_{ticker}'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length, ticker):
    sequences = []
    labels = []
    indices = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length].values)
        labels.append(data.iloc[i + seq_length][f'Next_Close_{ticker}'])
        start_date = data.index[i].strftime('%Y-%m-%d')
        end_date = data.index[i + seq_length - 1].strftime('%Y-%m-%d')
        indices.append(f"{start_date} - {end_date}")

    sequences_df = pd.DataFrame({
        'Sequence': sequences,
        f'Next_Close_{ticker}': labels
    }, index=indices)

    return sequences_df

def plot_histogram(df, column_name, title):
    fig, ax = plt.subplots()
    sns.histplot(df[column_name], kde=True, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_corr_matrix(df, ticker):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix[[f'Next_Close_{ticker}']], annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
    ax.set_title(f'Matriz de correlación para {ticker}')
    st.pyplot(fig)

def plot_training_loss(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    ax.plot(history.history['val_loss'], label='Pérdida de Validación')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Pérdida')
    ax.set_title('Evolución de la Pérdida durante el Entrenamiento')
    ax.legend()
    st.pyplot(fig)

def plot_predictions_vs_real(y_test, y_pred, ticker):
    fig, ax = plt.subplots()
    ax.plot(y_test, label='Valor Real')
    ax.plot(y_pred, label='Predicción')
    ax.set_xlabel('Observaciones')
    ax.set_ylabel('Precio de Cierre')
    ax.set_title(f'Comparación del Valor Real vs Predicción del Modelo para {ticker} en el Conjunto de Prueba')
    ax.legend()
    st.pyplot(fig)

def plot_scatter(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Predicho')
    ax.set_title('Gráfico de Dispersión de Valores Reales vs Predichos')
    st.pyplot(fig)

def plot_real_prices_full(seq_length, data, model, scaler_y, ticker):
    # Crear secuencias y etiquetas a partir de los datos completos
    sequences_df = create_sequences(data, seq_length, ticker)
    
    # Convertir la columna 'Sequence' a arreglos numpy y verificar los tipos de datos
    X_full = np.array(sequences_df['Sequence'].tolist(), dtype=np.float32)
    
    # Verificar tipos de datos
    print("Tipos de datos de las secuencias:", X_full.dtype)
    
    # Aplanar las secuencias
    X_full_flat = np.array([x.flatten() for x in X_full], dtype=np.float32)
    
    # Verificar tipos de datos
    print("Tipos de datos de las secuencias aplanadas:", X_full_flat.dtype)
    
    # Realizar predicciones
    y_pred_full = model.predict(X_full_flat)
    
    # Desescalar las predicciones
    y_pred_full_descaled = scaler_y.inverse_transform(y_pred_full)
    
    # Desescalar los valores reales
    y_real_full_descaled = scaler_y.inverse_transform(data.iloc[seq_length:][f'Next_Close_{ticker}'].values.reshape(-1, 1))
    
    # Crear un DataFrame con los resultados
    result_df = pd.DataFrame({
        'Fecha': data.index[seq_length:],
        'Valor Real': y_real_full_descaled.flatten(),
        'Predicción': y_pred_full_descaled.flatten()
    })
    
    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result_df['Fecha'], result_df['Valor Real'], label='Valor Real')
    ax.plot(result_df['Fecha'], result_df['Predicción'], label='Predicción Modelo MLP')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de Cierre')
    ax.set_title(f'Comparación del Valor Real vs Predicción del Modelo MLP para {ticker}')
    ax.legend()
    st.pyplot(fig)
    
    return result_df


def predict_next_day_price(model, data, seq_length, scaler_y, ticker):
    last_sequence = data.iloc[-seq_length:].values.reshape(1, seq_length * data.shape[1])
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler_y.inverse_transform(next_day_scaled.reshape(-1, 1))
    return next_day_price[0][0]

def app():
    st.title('Predicción de Precios con MLP')
    st.header('Configuración')

    # Parámetros del usuario
    ticker = st.selectbox('Ticker', ['BVN', 'FSM', 'SCCO'])
    start_date = st.date_input('Fecha de Inicio', pd.to_datetime('2018-01-01'))
    end_date = st.date_input('Fecha de Fin', pd.to_datetime('2022-12-31'))
    seq_length = 60
    epochs = 50
    batch_size = 32

    # Descargar y preparar los datos
    df = download_data(ticker, start_date, end_date)
    df = calculate_indicators(df, ticker)
    
    st.write(f"Se tienen los siguientes datos para {ticker}:")
    st.dataframe(df.head())

    # Calcular RSI y MACD
    st.write(f'##Calculamos el RSI y MACD para {ticker}.')
    df[f'RSI_{ticker}'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df[f'MACD_{ticker}'] = macd.macd()
    df[f'MACD_signal_{ticker}'] = macd.macd_signal()
    df[f'MACD_diff_{ticker}'] = macd.macd_diff()

    plot_histogram(df, f'RSI_{ticker}', f'RSI_{ticker}')
    plot_histogram(df, f'MACD_{ticker}', f'MACD_{ticker}')
    plot_histogram(df, f'MACD_signal_{ticker}', f'MACD_signal_{ticker}')
    plot_histogram(df, f'MACD_diff_{ticker}', f'MACD_diff_{ticker}')

    st.write(f'##Datos con la variable objetivo Next_Close_{ticker}')
    st.dataframe(df)

    st.write(f'##Matriz de correlación de la variable objetivo Next_Close_{ticker}')
    plot_corr_matrix(df, ticker)

    st.write(f'##Dataframe con las variables seleccionadas para {ticker}')
    selected_columns = [
        f'Prev Close_{ticker}', f'Prev High_{ticker}', f'Prev Low_{ticker}', f'Prev Open_{ticker}', 
        f'SMA_50_{ticker}', f'EMA_50_{ticker}', f'BB_Middle_{ticker}', f'BB_Upper_{ticker}', 
        f'Avg Price_{ticker}', f'RSI_{ticker}', f'MACD_{ticker}', f'MACD_signal_{ticker}', 
        f'MACD_diff_{ticker}', f'Next_Close_{ticker}'
    ]
    df_final = df[selected_columns]
    df_final.dropna(inplace=True)
    st.dataframe(df_final)

    st.write('## Dataframe normalizado')
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    df_final.loc[:, selected_columns[:-1]] = scaler_X.fit_transform(df_final[selected_columns[:-1]])
    df_final[f'Next_Close_{ticker}'] = scaler_y.fit_transform(df_final[[f'Next_Close_{ticker}']])
    st.dataframe(df_final)

    st.write('## Crear secuencias para el entrenamiento y prueba')
    sequences_df = create_sequences(df_final, seq_length, ticker)
    # Convertir la columna 'Sequence' a una representación de cadena
    sequences_df['Sequence'] = sequences_df['Sequence'].apply(lambda x: x.tolist())
    st.dataframe(sequences_df.head())

    st.write('## Preparar datos para el modelo MLP')
    X = np.array(sequences_df['Sequence'].tolist())
    y = np.array(sequences_df[f'Next_Close_{ticker}'].tolist())
    X_flat = np.array([x.flatten() for x in X])
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    # Definir modelo MLP
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compilar modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Entrenar modelo
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=2
    )

    # Evaluar modelo
    loss = model.evaluate(X_test, y_test, verbose=0)
    rmse = np.sqrt(loss)
    mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
    
    st.write(f'## Evaluación del modelo')
    st.write(f'Pérdida en el conjunto de prueba (MSE): {loss:.8f}')
    st.write(f'RMSE en el conjunto de prueba: {rmse:.4f}')
    st.write(f'MAPE: {mape:.4f}')

    st.write('## Gráfico de pérdida durante el entrenamiento')
    plot_training_loss(history)

    st.write('## Comparación de valores reales vs predichos')
    y_pred = model.predict(X_test)
    plot_predictions_vs_real(y_test, y_pred, ticker)

    st.write('## Gráfico de dispersión de valores reales vs predichos')
    plot_scatter(y_test, y_pred)

    st.write(f'## Comparación del Valor Real vs Predicción del Modelo MLP para {ticker}')
    result_df = plot_real_prices_full(seq_length, df_final, model, scaler_y, ticker)
    st.dataframe(result_df)

    st.write('## Predicción del precio del siguiente día')
    next_day_price = predict_next_day_price(model, df_final, seq_length, scaler_y, ticker)
    st.write(f'El precio predicho del siguiente día para {ticker} es: {next_day_price:.2f}')