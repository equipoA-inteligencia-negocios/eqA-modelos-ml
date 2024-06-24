import yfinance as yf
import pandas as pd
import ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from despliegue.semana11.modelo_lstm import plot_moving_average, descargar_datos, preprocesar_datos
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import r2_score

def separar_datos(datos, ticker):
    X = datos[[f'Open_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Close_{ticker}', f'Adj Close_{ticker}', f'Prev Close_{ticker}', f'Prev High_{ticker}',
                f'Prev Low_{ticker}', f'Prev Open_{ticker}', f'SMA_50_{ticker}', f'EMA_50_{ticker}',f'BB_Middle_{ticker}', f'BB_Upper_{ticker}', f'Avg Price_{ticker}']]
    y = datos[[f'Next_Close_{ticker}']]
    # Dividimos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_evaluate_svr(X_train, y_train, X_test, kernel='poly'):
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test).reshape(-1, 1)
    return svr, y_pred_svr

def train_evaluate_mlp(X_train, y_train, X_test, hidden_layer_sizes=(100,), max_iter=500, random_state=42):
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test).reshape(-1, 1)
    return mlp, y_pred_mlp

def train_evaluate_base_radial(X_train, y_train, X_test, kernel='rbf'):
  rbf = SVR(kernel=kernel)
  rbf.fit(X_train, y_train)
  y_pred_rbf = rbf.predict(X_test).reshape(-1, 1)
  return rbf, y_pred_rbf

def create_dataset(mlp, rbf, svr, ticker,
                   X_train, y_train, X_test, y_test,
                   y_pred_mlp, y_pred_rbf, y_pred_svr):
  hybrid_data_train = pd.DataFrame()
  hybrid_data_test = pd.DataFrame()
  hybrid_data_train.index = X_train.index

  hybrid_data_train['MLP_Pred'] = mlp.predict(X_train)
  hybrid_data_train['RBF_Pred'] = rbf.predict(X_train)
  hybrid_data_train['SVR_Pred'] = svr.predict(X_train)
  hybrid_data_train[f'Next_Close_{ticker}'] = y_train

  hybrid_data_test.index = X_test.index
  hybrid_data_test['MLP_Pred'] = y_pred_mlp
  hybrid_data_test['RBF_Pred'] = y_pred_rbf
  hybrid_data_test['SVR_Pred'] = y_pred_svr
  hybrid_data_test[f'Next_Close_{ticker}'] = y_test

  hybrid_data_train = pd.concat([X_train,hybrid_data_train], axis=1)
  hybrid_data_test = pd.concat([X_test, hybrid_data_test], axis=1)
  hybrid_data_train.dropna(inplace=True)
  hybrid_data_test.dropna(inplace=True)

  return hybrid_data_train, hybrid_data_test

def create_variables(hybrid_data_train, hybrid_data_test, ticker):
  X_hybrid_train = hybrid_data_train.drop(columns=[f'Next_Close_{ticker}'])
  y_hybrid_train = hybrid_data_train[f'Next_Close_{ticker}']

  X_hybrid_test = hybrid_data_test.drop(columns=[f'Next_Close_{ticker}'])
  y_hybrid_test = hybrid_data_test[f'Next_Close_{ticker}']

  return X_hybrid_train, y_hybrid_train, X_hybrid_test, y_hybrid_test

def train_evaluate_ann(X_hybrid_train, y_hybrid_train, X_hybrid_test, y_hybrid_test):
  # Crear el modelo de Red Neuronal Artificial
  model = Sequential()
  model.add(Input(shape=(X_hybrid_train.shape[1],)))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1))

  # Compilar el modelo
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Entrenar el modelo
  model.fit(X_hybrid_train, y_hybrid_train, epochs=50, batch_size=32, validation_split=0.2)

  # Predecir con el modelo final
  y_final_pred = model.predict(X_hybrid_test).reshape(-1, 1)

  # Evaluar el modelo
  mse = mean_squared_error(y_hybrid_test, y_final_pred)
  print(f"Mean Squared Error del modelo final: {mse}")

  return model, y_final_pred

def evaluate_models(model, y_test, y_pred):
  mape = mean_absolute_percentage_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  return mape, rmse

def resultados(y_hybrid_test, y_pred_ann):
  # Evaluación del modelo híbrido
  r2 = r2_score(y_hybrid_test, y_pred_ann)
  print(f'Modelo Híbrido R2: {r2}')

  resultados = pd.DataFrame({
      'Fecha': y_hybrid_test.index,
      'Valor Real': y_hybrid_test,
      'Predicción Modelo Híbrido': y_pred_ann.flatten()
  })
  return resultados

def plot_predicciones(resultados):
  plt.figure(figsize=(14, 7))
  plt.plot(resultados['Fecha'], resultados['Valor Real'], label='Valor Real')
  plt.plot(resultados['Fecha'], resultados['Predicción Modelo Híbrido'], label='Predicción Modelo Híbrido')
  plt.xlabel('Fecha')
  plt.ylabel('Precio de Cierre')
  plt.title('Comparación del Valor Real vs Predicción del Modelo Híbrido')
  plt.legend()
  return st.pyplot(plt)

def app():
    st.title("Modelo de Predicción de Precios de Acciones")
    
    st.write("""
    ### Bienvenido a la aplicación de predicción de precios de acciones. 
    Esta aplicación utiliza un modelo híbrudo para predecir los precios de cierre de acciones y determinar la tendencia del mercado.
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
        
        # datos = preprocesar_datos(datos, ticker, fecha_inicio, fecha_fin)
        datos, scaler = preprocesar_datos(datos, ticker, fecha_inicio, fecha_fin)

        # Separar los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = separar_datos(datos, ticker)        


        # Ploteo de la media móvil
        st.subheader("Media Móvil de los Precios Reales")
        plot_moving_average(datos, ticker)

        # Entrenar modelo
        st.subheader("Entrenamiento del Modelo")
        st.write("""
        El modelo se entrena utilizando un modelo híbrido. Se combinan las predicciones de un modelo de Red Neuronal Artificial (MLP), un modelo de Máquinas de Soporte Vectorial (SVR) y un modelo de Regresión Basado en Funciones Radiales (RBF). 
        Este modelo es capaz de predecir los precios de cierre de las acciones y determinar la tendencia del mercado.
        """)
        
        svr, y_pred_svr = train_evaluate_svr(X_train, y_train, X_test)
        mlp, y_pred_mlp = train_evaluate_mlp(X_train, y_train, X_test)
        rbf, y_pred_rbf = train_evaluate_base_radial(X_train, y_train, X_test)

        # Crear el dataset híbrido
        hybrid_data_train, hybrid_data_test = create_dataset(mlp, rbf, svr, ticker, X_train, y_train, X_test, y_test, y_pred_mlp, y_pred_rbf, y_pred_svr)
        X_hybrid_train, y_hybrid_train, X_hybrid_test, y_hybrid_test = create_variables(hybrid_data_train, hybrid_data_test, ticker)
        ann, y_final_pred = train_evaluate_ann(X_hybrid_train, y_hybrid_train, X_hybrid_test, y_hybrid_test)
        
        # Realizar predicciones
        st.subheader("Predicciones del Modelo")
        st.write("""
        Después de entrenar el modelo, se utilizan los datos de prueba para realizar pruebas sobre el modelo y evaluar su rendimiento.
        """)
        mape, rmse = evaluate_models(svr, y_test, y_pred_svr)
        st.write(f"SVR MAPE: {mape}")
        st.write(f"SVR RMSE: {rmse}")
        mape, rmse =  evaluate_models(mlp, y_test, y_pred_mlp)
        st.write(f"MLP MAPE: {mape}")
        st.write(f"MLP RMSE: {rmse}")
        mape, rmse = evaluate_models(rbf, y_test, y_pred_rbf)
        st.write(f"RBF MAPE: {mape}")
        st.write(f"RBF RMSE: {rmse}")
        mape, rmse = evaluate_models(ann, y_test, y_final_pred)
        st.write(f"Modelo Híbrido MAPE: {mape}")
        st.write(f"Modelo Híbrido RMSE: {rmse}")

        resultados_pred = resultados(y_hybrid_test, y_final_pred)
        st.subheader(f"Precio de Cierre del día siguiente real para la minera {ticker} y predicciones del modelo híbrido")
        st.dataframe(resultados_pred)

        # Predicción del precio del día siguiente
        st.subheader("Gráfico de Predicción del Precio del Día Siguiente")
        plot_predicciones(resultados_pred)