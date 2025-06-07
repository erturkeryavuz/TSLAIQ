import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(csv_path, lookback=60):
    df = pd.read_csv(csv_path).dropna()
    features = ["Close", "RSI", "EMA20", "EMA50", "MACD", "OBV"]

    data = df[features].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i - lookback:i])
        y.append(data_scaled[i][0])  # 'Close' hedef

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

# 📊 Test çalıştırmak için (isteğe bağlı)
if __name__ == "__main__":
    X, y, scaler = prepare_lstm_data("data/tsla_data.csv")
    print("🔢 X shape:", X.shape)
    print("🎯 y shape:", y.shape)
    print("🧠 Toplam örnek sayısı:", len(y))

def create_features_from_df(df):
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
