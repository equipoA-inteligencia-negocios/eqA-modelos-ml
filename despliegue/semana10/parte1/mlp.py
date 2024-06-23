from despliegue.semana10.parte1.preprocesamiento import preprocesar
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def app():
    st.title("Modelo de Predicción de IGBVL")
    
    st.write("""
    ### Bienvenido a la aplicación de predicción de IGBVL. 
    Esta aplicación utiliza un modelo MLP para predecir el IGBVL.
    """)
    
    df = pd.read_csv('despliegue/semana10/parte1/Datos.csv')
    
    st.write("""
    Se tienen los siguientes datos:
    """)
    
    st.dataframe(df)
    
    df_final, scaler_y = preprocesar(df)

    X = df_final[['Cooper', 'GOLD', 'Dow', 'IGBVL']]
    y = df_final['Next_IGBVL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configuramos el modelo MLP (probamos varias configuraciones, pero estas fueron las mejores)
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=800, solver='adam', random_state=42)

    # Entrenamos el modelo
    mlp_regressor.fit(X_train, y_train)

    # Realizamos la predicción
    y_pred = mlp_regressor.predict(X_test)
    st.subheader('Evaluación del modelo:')
    # Calculamos el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred) 
    # Calculamos el error absoluto medio
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f'El error cuadrático medio es: {mse}')
    st.write(f'El error absoluto medio es: {mae}')

    st.subheader('Predicciones vs valores reales')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    st.pyplot(plt)

    st.subheader('Predicción del día siguiente')
    last_value = X.iloc[-1]
    last_day_pred = mlp_regressor.predict(last_value.values.reshape(1, -1))
    last_day_noscaled = scaler_y.inverse_transform(last_day_pred.reshape(-1, 1)).flatten()
    st.write(f'Se espera que el IGBVL del día siguiente sea de: {last_day_noscaled[0]:.4f}')