# 📈 Price Direction Prediction with LSTM – ICICIBANK.NS (NSE, 5-Minute Data)

This project uses a **Long Short-Term Memory (LSTM)** neural network to predict the **next 5-minute price direction** of **ICICIBANK.NS** based on short-term market microstructure features.

Rather than predicting exact prices, the model focuses on a **classification problem**:
> Will the mid-price go **up (1)** or **down (0)** in the next bar?

---

## 🧠 Problem Setup

- **Instrument:** ICICI Bank (ICICIBANK.NS) – NSE
- **Data:** 5-minute OHLCV candles (last 5 trading days, via `yfinance`)
- **Target:** 1 if next mid-price > current mid-price, else 0
- **Model:** LSTM neural network (TensorFlow / Keras)

---

## 🏗 Features (Microstructure Inspired)

For each bar we build:

- `mid_price` = (High + Low) / 2  
- `spread`    = High − Low  
- `returns`   = % change of `mid_price`  

Then we create **rolling sequences** of length 20:

- Input shape to LSTM: `(20 timesteps × 3 features)`  

---

## 🧮 Model Architecture

```text
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dense(1, activation="sigmoid")
