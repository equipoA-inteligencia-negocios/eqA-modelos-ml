import yfinance as yf
import ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def importar_igbvl(fecha_inicio, fecha_fin):
    IGBVL_data = yf.download('^SPBLPGPT', start = fecha_inicio, end = fecha_fin)
    # Removiendo columna 'Volume' y 'Adj Close'
    IGBVL_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    IGBVL_data.columns += "_IGBVL"
    return IGBVL_data

def importar_dji(fecha_inicio, fecha_fin):
    DJI_data = yf.download('^DJI', start = fecha_inicio, end = fecha_fin)
    # Removiendo columna 'Volume' y 'Adj Close'
    DJI_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    DJI_data.columns += "_DJI"
    return DJI_data

def importar_nasdaq(fecha_inicio, fecha_fin):
    NASDAQ_data = yf.download('^IXIC', start = fecha_inicio, end = fecha_fin)
    # Removiendo columna 'Volume' y 'Adj Close'
    NASDAQ_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    NASDAQ_data.columns += "_NASDAQ"
    return NASDAQ_data

def importar_dolar(fecha_inicio, fecha_fin):
    PEN_X_data = yf.download('PEN=X', start = fecha_inicio, end = fecha_fin)
    # Removiendo columna 'Volume' y 'Adj Close'
    PEN_X_data.drop( ['Adj Close', 'Volume'] , axis=1, inplace=True)
    PEN_X_data.columns += "_PEN_X"
    return PEN_X_data

def importar_oro(fecha_inicio, fecha_fin):
    GLD_data = yf.download('GLD', start = fecha_inicio, end = fecha_fin)
    # Removiendo columna 'Volume' y 'Close'
    GLD_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    GLD_data.columns += "_GLD"
    return GLD_data

def importar_plata(fecha_inicio, fecha_fin):
    SIF_data = yf.download('SI=F', start = fecha_inicio, end = fecha_fin)
    # Removiendo columna 'Volume' y 'Adj Close'
    SIF_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    SIF_data.columns += "_SIF"
    return SIF_data

def importar_cobre(fecha_inicio, fecha_fin):
    HGF_data = yf.download('HG=F', start = fecha_inicio, end = fecha_fin)
    # Removiendo columna 'Volume' y 'Close'
    HGF_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    HGF_data.columns += "_HGF"
    return HGF_data

def importar_zinc(fecha_inicio, fecha_fin):
    T09_ZINC = yf.download('ZINC.L', start=fecha_inicio, end=fecha_fin)
    # Removiendo columna 'Volume' y 'Close'
    T09_ZINC.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    T09_ZINC.columns += "_ZINC"
    return T09_ZINC

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

def mostrar_correlacion(df):
    correlation_matrix = df.corr(method='pearson')
    plt.figure(figsize=(30, 27))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación de Pearson')
    plt.show()

def combinar_df(df, fecha_inicio, fecha_fin):
    IGBVL_data = importar_igbvl(fecha_inicio, fecha_fin)
    DJI_data = importar_dji(fecha_inicio, fecha_fin)
    NASDAQ_data = importar_nasdaq(fecha_inicio, fecha_fin)
    PEN_X_data = importar_dolar(fecha_inicio, fecha_fin)
    GLD_data = importar_oro(fecha_inicio, fecha_fin)
    SIF_data = importar_plata(fecha_inicio, fecha_fin)
    HGF_data = importar_cobre(fecha_inicio, fecha_fin)
    T09_ZINC = importar_zinc(fecha_inicio, fecha_fin)
    new_df = pd.merge(df, IGBVL_data, on='Date')
    new_df = pd.merge(new_df, DJI_data, on='Date')
    new_df = pd.merge(new_df, NASDAQ_data, on='Date')
    new_df = pd.merge(new_df, PEN_X_data, on='Date')
    new_df = pd.merge(new_df, GLD_data, on='Date')
    new_df = pd.merge(new_df, SIF_data, on='Date')
    new_df = pd.merge(new_df, HGF_data, on='Date')
    new_df = pd.merge(new_df, T09_ZINC, on='Date')
    return new_df