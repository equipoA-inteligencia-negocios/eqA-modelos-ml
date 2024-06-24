import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from despliegue.semana10.parte1.lstm_template import split_data
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def build(seq_length, X_train):
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(50, activation='relu', input_shape=(seq_length, 5)))
    model_rnn.add(Dense(1))
    model_rnn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model_rnn

def train(model_rnn, X_train_rnn, y_train_rnn):
    model_rnn.fit(
    X_train_rnn,       # Datos de entrada para el entrenamiento
    y_train_rnn,       # Etiquetas correspondientes
    epochs=100,              # Número de veces que el modelo verá el conjunto completo de datos de entrenamiento
    validation_split=0.3,   # Porcentaje del conjunto de entrenamiento utilizado para la validación
    )

def evaluate(model_rnn, X_test_rnn, y_test_rnn):
    loss_rnn = model_rnn.evaluate(X_test_rnn, y_test_rnn, verbose=0)
    st.write(f'Pérdida en el conjunto de prueba (MSE): {loss_rnn} ({loss_rnn:.8f})')

    # Calculamos el RMSE en el conjunto de prueba
    rmse_rnn = np.sqrt(loss_rnn)
    st.write(f'RMSE en el conjunto de prueba: {rmse_rnn}')

def predict_test(model_rnn, X_test_rnn, y_test_rnn, y_test_indexes_rnn, scaler_y):
    # Hacemos predicciones con el modelo entrenado
    y_pred_rnn = model_rnn.predict(X_test_rnn)
    # Desescalar los valores reales y las predicciones
    y_test_rnn_descaled = scaler_y.inverse_transform(y_test_rnn.reshape(-1, 1))
    y_pred_rnn_descaled = scaler_y.inverse_transform(y_pred_rnn.reshape(-1, 1))

    results_rnn_descaled = pd.DataFrame({
    'Fecha': y_test_indexes_rnn,
    'Valor Real': y_test_rnn_descaled.flatten(),
    'Predicción Modelo RNN': y_pred_rnn_descaled.flatten()
    })

    # Ordenar el DataFrame por Fecha
    results_rnn_descaled.sort_values(by='Fecha', inplace=True)

    # Visualización de resultados
    plt.figure(figsize=(14, 7))
    plt.plot(results_rnn_descaled['Fecha'], results_rnn_descaled['Valor Real'], label='Valor Real')
    plt.plot(results_rnn_descaled['Fecha'], results_rnn_descaled['Predicción Modelo RNN'], label='Predicción Modelo RNN')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title('Comparación del Valor Real vs Predicción del Modelo RNN')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Dataframe de predicciones en el conjunto de datos completo")
    st.dataframe(results_rnn_descaled)