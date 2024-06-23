import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from ta.momentum import RSIIndicator # RSI
from ta.trend import MACD # MACD
import seaborn as sns
import matplotlib.pyplot as plt
    
def preprocesar(df):
    df = df.dropna()
    
    label_encoder = LabelEncoder()
    df.loc[:, 'Dir'] = label_encoder.fit_transform(df['Dir'])
    
    # Calculamos EL RSI Y MACD. Estos se aplican principalmente a las columnas que representan un precio de cierre o el precio de un activo, por lo que lo aplicaremos a IGBVL
    
    st.write('Calculamos EL RSI Y MACD. Estos se aplican principalmente a las columnas que representan un precio de cierre o el precio de un activo, por lo que lo aplicaremos a IGBVL.')
   
    df.loc[:,'RSI'] = RSIIndicator(df['IGBVL']).rsi()
    df.head(20)
    # En este caso el RSIIndicator utiliza una ventana de 14 por defecto, por esa razón es que los 13 primeros valores están vacíos dado que el indicador necesita un número determinado
    # de datos previos para el cálculo

    """### MACD"""

    # MACD
    macd = MACD(df['IGBVL'])
    df.loc[:,'MACD'] = macd.macd()
    df.loc[:,'MACD_signal'] = macd.macd_signal()
    df.loc[:,'MACD_diff'] = macd.macd_diff()

    # En el caso de la imputación, debemos saber si es mejor imputar la media (distribución normal) o la mediana (sesgada)
    
    st.write('## RSI')
    sns.histplot(df['RSI'], kde=True)
    # plt.show()
    st.pyplot(plt)

    st.write('## MACD')
    sns.histplot(df['MACD'], kde=True)
    st.pyplot(plt)

    st.write('## MACD_signal')
    sns.histplot(df['MACD_signal'], kde=True)
    st.pyplot(plt)

    st.write('## MACD_diff')
    sns.histplot(df['MACD_diff'], kde=True)
    st.pyplot(plt)

    """# B) VARIABLE OBJETIVO"""

    df['Next_IGBVL'] = df['IGBVL'].shift(-1)

    """# C) LIMPIEZA DE DATOS"""

    # Eliminamos la columna 'Nro', ya que ya está representada por el índice
    df = df.drop('Nro', axis=1)

    # Eliminamos los valores nulos
    df.dropna(inplace=True)
    df.isnull().sum()
    
    st.write('## Datos con la variable objetivo Next_IGBVL')
    st.dataframe(df)

    """# D) MATRIZ DE CORRELACIÓN"""

    corr_matrix = df.corr()
    plt.figure(figsize=(16,14))
    sns.heatmap(corr_matrix[['Next_IGBVL']], annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Matriz de correlación')
    st.write('## Matriz de correlación de la variable objetivo Next_IGBVL')
    st.pyplot(plt)

    st.write('## Dataframe con las variables seleccionadas')
    # Creamos un dataframe con estas variables y también con la variable objetivo
    df_final = df[['Cooper', 'GOLD', 'Dow', 'IGBVL', 'Next_IGBVL']]
    st.dataframe(df_final)

    """# E) NORMALIZACIÓN"""

    from sklearn.preprocessing import MinMaxScaler

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    columns_X = df_final.columns.drop('Next_IGBVL')
    columns_X

    # Normalizamos las variables independientes
    df_final.loc[:, columns_X] = scaler_X.fit_transform(df_final[columns_X])

    # Normalizamos la variable dependiente
    df_final.loc[:, 'Next_IGBVL'] = scaler_y.fit_transform(df_final['Next_IGBVL'].values.reshape(-1,1)).flatten()

    st.write('## Dataframe normalizado')
    st.dataframe(df_final)
    
    return df_final