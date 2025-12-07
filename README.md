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

git clone https://github.com/<your-username>/Order-Book-Direction-LSTM-ICICIBANK.git
cd Order-Book-Direction-LSTM-ICICIBANK
pip install -r requirements.txt   # or install manually: yfinance, pandas, numpy, scikit-learn, tensorflow, matplotlib
jupyter notebook


You can also add your accuracy plot as `images/accuracy.png` and reference it in the README.

---

## 3️⃣ LinkedIn post for this LSTM project

Here’s a ready caption you can post:

```text
🚀 New Quant + Deep Learning Project!

I built an LSTM-based model to predict short-term price direction for **ICICIBANK.NS** using 5-minute data from NSE.

What the model does:
• Uses microstructure-style features (mid-price, spread, short-term returns)
• Builds rolling sequences of the last 20 bars
• Predicts whether the next bar’s mid-price will move UP or DOWN
• Achieves around **60.6% accuracy** on the test set (vs 50% random)

Tech Stack:
Python · Pandas · NumPy · scikit-learn · TensorFlow/Keras · yFinance · Jupyter

GitHub Repo 🔗:
<your repo link here>

Key learnings:
• Even simple LSTM architectures can extract signal from noisy intraday data  
• Feature engineering (mid-price, spread, returns) matters a lot  
• This type of directional edge is the starting point for execution algos & HFT models

Next steps:
• Turn predictions into a trading strategy with transaction costs  
• Add order-imbalance & volume features  
• Experiment with 1D-CNNs / Transformers

Happy to connect with people working in Quant / HFT / Trading Tech 🚀

#quant #deeplearning #lstm #nse #trading #hft #python #machinelearning
