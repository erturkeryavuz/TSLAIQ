import os
import time
import numpy as np
from datetime import datetime
import webbrowser
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Lambda
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from veriHazirlaLstm import prepare_lstm_data
from tensorboard import program
import tensorflow as tf  # 💡 Bu satır eksikse Lambda hatası verir

def launch_tensorboard(log_dir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f"🚀 TensorBoard başlatıldı: {url}")
    webbrowser.open(url)

X, y, scaler = prepare_lstm_data("data/tsla_data.csv", lookback=60)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

def build_attention_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inputs)

    score = Dense(64, activation='tanh')(lstm_out)
    attention_weights = Dense(1)(score)
    attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=1), name="attention_softmax")(attention_weights)


    context_vector = Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1),
        output_shape=(64,)
    )([attention_weights, lstm_out])

    last_lstm_output = Lambda(
        lambda x: x[:, -1, :],
        output_shape=(64,)
    )(lstm_out)

    concat = Concatenate()([context_vector, last_lstm_output])
    dropout = Dropout(0.3)(concat)
    output = Dense(1)(dropout)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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

os.makedirs("saved_models", exist_ok=True)
best_loss, best_file = get_best_existing_loss()
print(f"📁 En iyi mevcut val_loss: {best_loss:.5f} | Model: {best_file or 'Bulunamadı'}")

model = build_attention_lstm_model((X_train.shape[1], X_train.shape[2]))

log_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

start_time = time.time()
max_duration = 20 * 60
epoch_counter = 0
tensorboard_started = False

while True:
    elapsed = time.time() - start_time
    if elapsed > max_duration:
        print(f"\n⏱️ 20 dakika sınırına ulaşıldı. Eğitim tamamlandı.")
        break

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=32,
        verbose=1,
        callbacks=[
            EarlyStopping(patience=7, restore_best_weights=True),
            lr_scheduler,
            tensorboard_cb
        ]
    )

    if not tensorboard_started:
        launch_tensorboard(log_dir)
        tensorboard_started = True

    epoch_counter += 1
    val_loss = history.history["val_loss"][-1]
    print(f"⏳ {int(elapsed)}s | Epoch {epoch_counter} | val_loss = {val_loss:.5f}")

    with open("train_log.txt", "a") as f:
        f.write(f"{datetime.now()} | Epoch {epoch_counter} | val_loss: {val_loss:.5f}\n")

    if val_loss < best_loss:
        best_loss = val_loss
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_models/best_val{val_loss:.5f}_{timestamp}.keras"
        model.save(filename)
        print(f"✅ Yeni en iyi model kaydedildi: {filename}")
