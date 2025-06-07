import os
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model
import keras
from veriHazirlaLstm import prepare_lstm_data
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import Lambda

# TensorFlow fonksiyonları tanıtılsın
keras.config.enable_unsafe_deserialization()

DATA_PATH = "data/tsla_data.csv"
LOOKBACK = 60
FUTURE_DAYS = 7
MODEL_DIR = "saved_models"

# 🔍 En iyi modeli bul
def get_best_model_path():
    best_loss = float('inf')
    best_model = None
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("best_val") and fname.endswith(".keras"):
            try:
                val = float(fname.split("best_val")[1].split("_")[0])
                if val < best_loss:
                    best_loss = val
                    best_model = fname
            except:
                continue
    return os.path.join(MODEL_DIR, best_model) if best_model else None

# 📈 Gelecek günler için tahmin üret
def predict_future_price(model, df, scaler, lookback, future_days):
    features = ["Close", "RSI", "EMA20", "EMA50", "MACD", "OBV"]
    data = df[features].values
    data_scaled = scaler.fit_transform(data)

    last_seq = data_scaled[-lookback:]
    predictions = []

    for _ in range(future_days):
        input_seq = last_seq[-lookback:].reshape(1, lookback, len(features))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        next_row = last_seq[-1].copy()
        next_row[0] = pred_scaled  # Sadece Close tahmin ediliyor
        last_seq = np.vstack([last_seq, next_row])
        predictions.append(pred_scaled)

    predictions = scaler.inverse_transform(np.column_stack([predictions] * len(features)))[:, 0]
    return predictions

# 🧠 AL / SAT sinyali üret
def generate_signal(current_price, predicted_prices):
    future_max = np.max(predicted_prices)
    future_min = np.min(predicted_prices)
    future_mean = np.mean(predicted_prices)

    if future_max > current_price * 1.05:
        return "AL", f"📈 AL — 1 hafta içinde en yüksek tahmin: ${future_max:.2f}"
    elif future_min < current_price * 0.95:
        return "SAT", f"📉 SAT — 1 hafta içinde düşüş bekleniyor: ${future_min:.2f}"
    else:
        return "BEKLE", f"⏸️ BEKLE — Tahminler kararsız: ${future_mean:.2f}"

# 🚀 Ana fonksiyon
def main():
    print("\n🚀 Tesla fiyat tahmini başlatıldı...")
    model_path = get_best_model_path()
    if not model_path:
        print("⛔ En iyi model bulunamadı.")
        return

    df = pd.read_csv(DATA_PATH).dropna()
    X, y, scaler = prepare_lstm_data(DATA_PATH, lookback=LOOKBACK)

    model = load_model(
        model_path,
        compile=False,
        custom_objects={
            'tf': tf,
            'softmax': tf.nn.softmax,
            'Lambda': Lambda
        }
    )

    current_price = float(df["Close"].iloc[-1])
    predicted_prices = predict_future_price(model, df, scaler, LOOKBACK, FUTURE_DAYS)

    signal, message = generate_signal(current_price, predicted_prices)

    print(f"\n📊 Anlık Tesla fiyatı: ${current_price:.2f}")
    print(f"{message}")
    print(f"📢 Karar: {signal}")

    with open("prediction_log.txt", "a") as f:
        f.write(f"{datetime.now()} | Fiyat: ${current_price:.2f} | Tahmin: {predicted_prices.tolist()} | Karar: {signal}\n")

if __name__ == "__main__":
    main()
