import os
import numpy as np
import time
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from veriHazirlaLstm import prepare_lstm_data

# 🔢 Veriyi hazırla
X, y, scaler = prepare_lstm_data("data/tsla_data.csv", lookback=60)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🧠 Model mimarisi
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 📦 En iyi model dosyasını bul
def get_best_existing_loss(model_dir="saved_models"):
    best_loss = float('inf')
    best_model_file = None
    if not os.path.exists(model_dir):
        return best_loss, None
    for fname in os.listdir(model_dir):
        if fname.startswith("best_val") and fname.endswith(".keras"):
            try:
                val = float(fname.split("best_val")[1].split("_")[0])
                if val < best_loss:
                    best_loss = val
                    best_model_file = fname
            except:
                continue
    return best_loss, best_model_file

# 🔁 Eğitim süreci
os.makedirs("saved_models", exist_ok=True)
best_loss, best_file = get_best_existing_loss()

print(f"📁 En iyi mevcut val_loss: {best_loss:.5f} | Model: {best_file or 'Bulunamadı'}")

model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

val_loss = history.history["val_loss"][-1]
print(f"🧠 Yeni model val_loss: {val_loss:.5f}")

# 💾 Kayıt işlemi (gelişme varsa)
if val_loss < best_loss:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_models/best_val{val_loss:.5f}_{timestamp}.keras"
    model.save(filename)
    print(f"✅ Yeni en iyi model kaydedildi: {filename}")
else:
    print("⛔ Yeni model mevcut en iyiden daha kötü. Kaydedilmedi.")
