from despliegue.semana10.parte1.preprocesamiento import preprocesar
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        # Acceso por índice con .iloc
        sequences.append(data.iloc[i:i + seq_length].values)
        # Añadir el precio del día siguiente como etiqueta
        labels.append(data.iloc[i + seq_length]['Next_IGBVL'])

    # Crear DataFrame de secuencias
    sequences_df = pd.DataFrame({
        'Sequence': sequences,
        'Next_IGBVL': labels
    })

    return sequences_df

def split_data(df, seq_length_lstm):
    sequences_df_lstm = create_sequences(df, seq_length_lstm)

    # Separar las secuencias y etiquetas
    # Dividir en conjuntos de entrenamiento y prueba
    train_df_lstm, test_df_lstm = train_test_split(sequences_df_lstm, test_size=0.2, random_state=42)

    # Separar las secuencias y etiquetas
    X_train_lstm = np.array(train_df_lstm['Sequence'].tolist())
    y_train_lstm = np.array(train_df_lstm['Next_IGBVL'].tolist())
    X_test_lstm = np.array(test_df_lstm['Sequence'].tolist())
    y_test_lstm = np.array(test_df_lstm['Next_IGBVL'].tolist())
    
    # Asegurar que mantenemos los índices
    y_test_indexes_lstm = test_df_lstm.index
    
    return X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, y_test_indexes_lstm

def build(seq_length_lstm, X_train_lstm):
    # Construimos el modelo LSTM
    model_lstm = Sequential()

    # Añadimos una capa LSTM con 50 unidades y función de activación ReLU
    model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length_lstm, X_train_lstm.shape[2])))
    # LSTM(50): Añade una capa LSTM con 50 neuronas o unidades LSTM.
    # activation='relu': Utiliza la función de activación ReLU (Rectified Linear Unit), que es una función no lineal común en redes neuronales que ayuda a manejar problemas de gradientes desvanecidos.
    # input_shape=(timesteps, n_features): Define la forma de la entrada. Aquí, 'timesteps' es el número de pasos de tiempo y 'n_features' es el número de características en cada paso de tiempo.

    # Añadir una capa densa con una unidad de salida
    model_lstm.add(Dense(1))
    # Añade una capa completamente conectada con 1 neurona. Esta es la capa de salida del modelo, que proporciona la predicción final.
    # No se especifica una función de activación en esta capa, por lo que se utiliza la activación lineal por defecto.
    # Esto es adecuado para problemas de regresión donde la salida es un valor continuo, como en este caso que queremos predecir un precio.

    # Compilación del modelo
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    # optimizer=Adam(learning_rate=0.001): Utiliza el optimizador Adam con una tasa de aprendizaje de 0.001
    # loss='mean_squared_error': Utiliza la función de pérdida de error cuadrático medio (MSE)

    return model_lstm

def train(model_lstm, X_train_lstm, y_train_lstm, epochs, batch_size, validation_split):
    # Entrenamos el modelo con 50 épocas y un tamaño de lote de 32
    model_lstm.fit(
        X_train_lstm,       # Datos de entrada para el entrenamiento
        y_train_lstm,       # Etiquetas correspondientes
        epochs=epochs,              # Número de veces que el modelo verá el conjunto completo de datos de entrenamiento
        batch_size=batch_size,          # Número de muestras que se procesan antes de actualizar el modelo
        validation_split=validation_split,   # Porcentaje del conjunto de entrenamiento utilizado para la validación
        verbose=2               # Modo de verbosidad; 2 para una salida detallada por época
    )
    
def evaluate(model_lstm, X_test_lstm, y_test_lstm):
    # Evaluamos la pérdida del modelo (MSE) en el conjunto de prueba
    loss_lstm = model_lstm.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    st.write(f'Pérdida en el conjunto de prueba (MSE): {loss_lstm} ({loss_lstm:.8f})')

    # Calculamos el RMSE en el conjunto de prueba
    rmse_lstm = np.sqrt(loss_lstm)
    st.write(f'RMSE en el conjunto de prueba: {rmse_lstm}')

def predict_test(model_lstm, X_test_lstm, y_test_lstm, y_test_indexes_lstm, scaler_y, version):
    # Hacemos predicciones con el modelo entrenado
    y_pred_lstm = model_lstm.predict(X_test_lstm)
    # Desescalar los valores reales y las predicciones
    y_test_lstm_descaled = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1))
    y_pred_lstm_descaled = scaler_y.inverse_transform(y_pred_lstm.reshape(-1, 1))

    results_lstm_descaled = pd.DataFrame({
        'Fecha': y_test_indexes_lstm,
        'Valor Real': y_test_lstm_descaled.flatten(),
        'Predicción Modelo LSTM': y_pred_lstm_descaled.flatten()
    })

    # Ordenar el DataFrame por Fecha
    results_lstm_descaled.sort_values(by='Fecha', inplace=True)

    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(results_lstm_descaled['Fecha'], results_lstm_descaled['Valor Real'], label='Valor Real')
    plt.plot(results_lstm_descaled['Fecha'], results_lstm_descaled['Predicción Modelo LSTM'], label='Predicción Modelo LSTM')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title(f'Comparación del Valor Real vs Predicción del Modelo LSTM {version}')
    plt.legend()
    st.pyplot(plt)
    
# Crear secuencias para todo el conjunto de datos
def create_sequences_full(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length].values)
    return np.array(sequences)
    
def plot_real_prices_full(seq_length, data, model, scaler_y, version):
    # Configuración del modelo LSTM
    X_full = create_sequences_full(data, seq_length)

    # Realizar predicciones en todo el conjunto de datos
    y_pred_full = model.predict(X_full)

    # Desescalar las predicciones y los valores reales
    # full_indices = data.index[seq_length:]
    y_pred_full_descaled = scaler_y.inverse_transform(y_pred_full)
    y_real_full_descaled = scaler_y.inverse_transform(data.iloc[seq_length:][f'Next_IGBVL'].values.reshape(-1, 1))

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
    plt.title(f'Comparación del Valor Real vs Predicción del Modelo LSTM {version}')
    plt.legend()
    st.pyplot(plt)

    return results_full

# Predecir el precio del día siguiente
def predict_next_day_price(model, data, seq_length, scaler_y):
    # Obtener la última secuencia de datos escalados
    last_sequence = data.iloc[-seq_length:].values.reshape(1, seq_length, -1)

    # Hacer la predicción del siguiente día utilizando el modelo
    next_day_scaled = model.predict(last_sequence)

    # Desescalar la predicción
    next_day_price = scaler_y.inverse_transform(next_day_scaled.reshape(-1, 1))

    return next_day_price[0][0]