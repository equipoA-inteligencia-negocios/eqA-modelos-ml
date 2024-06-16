import ta
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib

# Función para descargar y procesar datos de Yahoo Finance
def descargar_datos(ticker, fechaInicio, fechaFin):
    data = yf.download(ticker, start=fechaInicio, end=fechaFin)
    data.drop(['Adj Close'], axis=1, inplace=True)
    data.columns += f"_{ticker}"
    return data

# Fechas de inicio y fin para la descarga de datos

fechaInicio = '2010-01-01'
fechaFin = '2023-01-01'

# Parte 1. Carga de Datos

SCCO_data = descargar_datos('SCCO', fechaInicio, fechaFin)
PEN_X_data = descargar_datos('PEN=X', fechaInicio, fechaFin)
IGBVL_data = yf.download('^SPBLPGPT', start=fechaInicio, end=fechaFin)
IGBVL_data.drop(['Adj Close'], axis=1, inplace=True)
IGBVL_data.columns += "_IGBVL"
DJI_data = descargar_datos('^DJI', fechaInicio, fechaFin)
NASDAQ_data = yf.download('^IXIC', start=fechaInicio, end=fechaFin)
NASDAQ_data.drop(['Adj Close'], axis=1, inplace=True)
NASDAQ_data.columns += "_NASDAQ"
HGF_data = yf.download('HG=F', start=fechaInicio, end=fechaFin)
HGF_data.drop(['Adj Close'], axis=1, inplace=True)
HGF_data.columns += "_HGF"
GLD_data = descargar_datos('GLD', fechaInicio, fechaFin)
SIF_data = yf.download('SI=F', start=fechaInicio, end=fechaFin)
SIF_data.drop(['Adj Close'], axis=1, inplace=True)
SIF_data.columns += "_SIF"
T09_ZINC_data = yf.download('ZINC.L', start=fechaInicio, end=fechaFin)
T09_ZINC_data.drop(['Adj Close'], axis=1, inplace=True)
T09_ZINC_data.columns += "_ZINC"


# Parte 2. Preprocesamiento de Datos
SCCO_data['Prev Close_SCCO'] = SCCO_data['Close_SCCO'].shift(1)
SCCO_data['Prev High_SCCO'] = SCCO_data['High_SCCO'].shift(1)
SCCO_data['Prev Low_SCCO'] = SCCO_data['Low_SCCO'].shift(1)
SCCO_data['Prev Open_SCCO'] = SCCO_data['Open_SCCO'].shift(1)
SCCO_data['RSI_SCCO'] = ta.momentum.RSIIndicator(SCCO_data['Close_SCCO']).rsi()
SCCO_data['Momentum_SCCO'] = SCCO_data['Close_SCCO'] - SCCO_data['Close_SCCO'].shift(10)
stoch = ta.momentum.StochasticOscillator(SCCO_data['High_SCCO'], SCCO_data['Low_SCCO'], SCCO_data['Close_SCCO'])
SCCO_data['Stoch_K_SCCO'] = stoch.stoch()
SCCO_data['Stoch_D_SCCO'] = stoch.stoch_signal()
SCCO_data['SMA_50_SCCO'] = ta.trend.SMAIndicator(SCCO_data['Close_SCCO'], window=50).sma_indicator()
SCCO_data['EMA_50_SCCO'] = ta.trend.EMAIndicator(SCCO_data['Close_SCCO'], window=50).ema_indicator()
SCCO_data['WilliamsR_SCCO'] = ta.momentum.WilliamsRIndicator(SCCO_data['High_SCCO'], SCCO_data['Low_SCCO'], SCCO_data['Close_SCCO']).williams_r()
SCCO_data['OBV_SCCO'] = ta.volume.OnBalanceVolumeIndicator(SCCO_data['Close_SCCO'], SCCO_data['Volume_SCCO']).on_balance_volume()
bollinger = ta.volatility.BollingerBands(SCCO_data['Close_SCCO'], window=20)
SCCO_data['BB_Middle_SCCO'] = bollinger.bollinger_mavg()
SCCO_data['BB_Upper_SCCO'] = bollinger.bollinger_hband()
SCCO_data['Avg Price_SCCO'] = (SCCO_data['Open_SCCO'] + SCCO_data['Close_SCCO']) / 2
SCCO_data['Amount_SCCO'] = SCCO_data['Close_SCCO'] * SCCO_data['Volume_SCCO']
SCCO_data['BIAS_SCCO'] = (SCCO_data['Close_SCCO'] - SCCO_data['SMA_50_SCCO']) / SCCO_data['SMA_50_SCCO']
SCCO_data['PVC_SCCO'] = SCCO_data['Volume_SCCO'].pct_change()
SCCO_data['AR_SCCO'] = SCCO_data['High_SCCO'].rolling(window=14).mean() / SCCO_data['Low_SCCO'].rolling(window=14).mean()
macd = ta.trend.MACD(SCCO_data['Close_SCCO'])
SCCO_data['MACD_SCCO'] = macd.macd()
SCCO_data['MACD_Signal_SCCO'] = macd.macd_signal()
SCCO_data['MACD_Hist_SCCO'] = macd.macd_diff()
SCCO_data['ROC_SCCO'] = ta.momentum.ROCIndicator(SCCO_data['Close_SCCO']).roc()
SCCO_data['PSY_SCCO'] = SCCO_data['Close_SCCO'].rolling(window=12).apply(lambda x: np.sum(x > x.shift(1)) / len(x))


# Combinando todos los datos
data_combined = pd.concat([SCCO_data, PEN_X_data, IGBVL_data, DJI_data, NASDAQ_data, HGF_data, GLD_data, SIF_data, T09_ZINC_data], axis=1)

# Escalando los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_combined)
data_scaled = pd.DataFrame(data_scaled, columns=data_combined.columns, index=data_combined.index)
data_scaled.dropna(inplace=True)

# Selección de variables por correlación
correlation_matrix = data_scaled.corr(method='pearson')
correlation_scco = correlation_matrix['Close_SCCO']
selected_variables = correlation_scco[correlation_scco.abs() > 0.8]

# Creando un nuevo dataset con las variables seleccionadas
selected_data = data_scaled[selected_variables.index]

# Dividir los datos en entrenamiento y prueba
X = selected_data.drop(columns=['Close_SCCO'])
y = selected_data['Close_SCCO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modelo LSTM
X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
lstm_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Otros modelos
svr_model = SVR(kernel='linear')
svr_model.fit(X_train, y_train)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
rbf_model = SVR(kernel='rbf')
rbf_model.fit(X_train, y_train)

# Creando dataset híbrido
hybrid_data_train = pd.DataFrame({
    'MLP_Pred': mlp_model.predict(X_train),
    'LSTM_Pred': lstm_model.predict(X_train_reshaped).flatten(),
    'SVR_Pred': svr_model.predict(X_train),
    'RBF_Pred': rbf_model.predict(X_train),
    'Real': y_train
})

hybrid_data_test = pd.DataFrame({
    'MLP_Pred': mlp_model.predict(X_test),
    'LSTM_Pred': lstm_model.predict(X_test_reshaped).flatten(),
    'SVR_Pred': svr_model.predict(X_test),
    'RBF_Pred': rbf_model.predict(X_test),
    'Real': y_test
})

# Modelo Híbrido
X_hybrid_train = hybrid_data_train.drop(columns=['Real'])
y_hybrid_train = hybrid_data_train['Real']
X_hybrid_test = hybrid_data_test.drop(columns=['Real'])
y_hybrid_test = hybrid_data_test['Real']

hybrid_model = SVR(kernel='linear')
hybrid_model.fit(X_hybrid_train, y_hybrid_train)

# Guardando el modelo híbrido
joblib.dump(hybrid_model, 'produccion/models/modelo_hibrido.pkl')

# Evaluación de los modelos
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mape, mse, r2

# Evaluar los modelos
models = {'SVR': svr_model, 'MLP': mlp_model, 'RBF': rbf_model, 'Hybrid': hybrid_model}
for model_name, model in models.items():
    if model_name == 'Hybrid':
        mape, mse, r2 = evaluate_model(model, X_hybrid_test, y_hybrid_test)
    else:
        mape, mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"{model_name} - MAPE: {mape:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")