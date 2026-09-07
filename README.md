# 📈 TSLAIQ — Tesla Stock Prediction & Signal Alerts

An LSTM-based time-series pipeline that predicts Tesla (TSLA) stock price movements and generates trading signal alerts.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## Overview

TSLAIQ pulls historical TSLA data, engineers technical indicators, trains LSTM models to forecast price movement, and turns predictions into buy/sell signal alerts.

## Pipeline

1. **Data ingestion** (`main.py`) — pulls TSLA price history via `yfinance`, incrementally updating a local CSV instead of re-downloading from scratch on each run
2. **Feature engineering** (`utils/indicators.py`) — adds technical indicators to the raw price data
3. **Model training** — two architectures are implemented and compared:
   - `train_lstm.py` — a standard LSTM model
   - `train_lstm_with_attention.py` — an LSTM variant with an attention mechanism

   Both use early stopping and checkpoint the best model by validation loss to `saved_models/`, with training runs logged to TensorBoard (`logs/`)
4. **Prediction & alerting** (`predict_and_signal.py`) — loads the best saved model and generates a trading signal alert from the latest data

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · yfinance

## Project Structure

```
TSLAIQ/
├── data/                          # cached TSLA price history
├── logs/                          # TensorBoard training logs
├── saved_models/                  # checkpointed .keras models (by validation loss)
├── utils/
│   └── indicators.py              # technical indicator feature engineering
├── main.py                        # data ingestion / update pipeline
├── veriHazirlaLstm.py             # data preparation for LSTM input
├── train_lstm.py                  # baseline LSTM training
├── train_lstm_with_attention.py   # attention-augmented LSTM training
└── predict_and_signal.py          # inference + signal alert generation
```

## Status

Personal research project exploring LSTM architectures for financial time-series prediction — not under active development.
