import yfinance as yf
import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt
from despliegue.adding_data import importar_plata, combinar_df
from sklearn.preprocessing import MinMaxScaler

def ltsm_pre(df_fsm):
    df_fsm_v2 = df_fsm[['Open_FSM', 'High_FSM', 'Low_FSM', 'Close_FSM', 'Adj Close_FSM', 'Prev Close_FSM', 'Prev High_FSM', 'Prev Low_FSM', 'Prev Open_FSM', 'SMA_50_FSM', 'EMA_50_FSM', 'BB_Middle_FSM', 'BB_Upper_FSM', 'Avg Price_FSM', 'Next_Close_FSM']]
    return df_fsm_v2

def ltsm_v4(df_fsm_v2, fecha_inicio, fecha_fin):
    SIF_data = importar_plata(fecha_inicio, fecha_fin)
    df_fsm_v4 = pd.merge(df_fsm_v2, SIF_data, on='Date')
    return df_fsm_v4

def final_pre(df_fsm_v4):
    scaler = MinMaxScaler(feature_range=(0, 1))
    columns_fsm = df_fsm_v4.columns
    df_fsm_v4[columns_fsm] = scaler.fit_transform(df_fsm_v4[columns_fsm])
    df_fsm_v4.dropna(inplace=True)
    return df_fsm_v4

def ltsm_final(df_fsm, fecha_inicio, fecha_fin):
    df_fsm_v2 = ltsm_pre(df_fsm)
    df_fsm_v3 = combinar_df(df_fsm_v2, fecha_inicio, fecha_fin)
    df_fsm_v4 = ltsm_v4(df_fsm_v3, fecha_inicio, fecha_fin)
    df_fsm_v5 = final_pre(df_fsm_v4)
    return df_fsm_v5