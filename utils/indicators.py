import pandas as pd
import ta

def add_technical_indicators(df):
    df = df.copy()

    # Close ve Volume sütunlarını Series olarak al
    close = df["Close"].squeeze()  # garanti tek boyut
    volume = df["Volume"].squeeze()

    # RSI
    rsi_indicator = ta.momentum.RSIIndicator(close=close)
    df["RSI"] = rsi_indicator.rsi()

    # EMA20 ve EMA50
    ema20 = ta.trend.EMAIndicator(close=close, window=20)
    ema50 = ta.trend.EMAIndicator(close=close, window=50)
    df["EMA20"] = ema20.ema_indicator()
    df["EMA50"] = ema50.ema_indicator()

    # MACD
    macd = ta.trend.MACD(close=close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    df["OBV"] = obv.on_balance_volume()

    return df
