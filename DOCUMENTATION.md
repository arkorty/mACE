### **1. Time Series Forecasting with LSTM/GRU (Deep Learning)**
The code uses **recurrent neural networks (RNNs)** with LSTM/GRU layers to predict future stock prices.

#### **Key Features:**
- **Sequence-based training** (using past `SEQUENCE_LENGTH` days to predict the next day)
- **Attention Mechanism** (optional)
- **Bidirectional RNNs** (optional)
- **Technical Indicators** as additional features (optional)

#### **Code Snippets:**
```python
# LSTM/GRU Model Architecture
model = Sequential()
if use_bidirectional:
    model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=input_shape))
else:
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
model.add(Dropout(DROPOUT_RATE))
model.add(LSTM(units=64, return_sequences=use_attention))
model.add(Dropout(DROPOUT_RATE))
if use_attention:
    model.add(AttentionLayer())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))
```

```python
# Feature Engineering (Technical Indicators)
def add_technical_indicators(df):
    df["SMA_5"] = df["Close"].rolling(window=5).mean()  # Simple Moving Average
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()  # Exponential Moving Average
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))  # Relative Strength Index
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()  # MACD
    return df
```

---

### **2. Moving Average Crossover Strategy (Technical Analysis)**
A classic trading strategy that generates **BUY/SELL signals** based on the crossover of short-term and long-term moving averages.

#### **Key Features:**
- **Short-term (5-day) vs. Long-term (20-day) moving averages**
- **Walk-forward validation** for backtesting
- **Accuracy metrics** (Precision, Recall, F1-score)

#### **Code Snippets:**
```python
# Moving Average Crossover Strategy
def moving_average_strategy(data, short_window=5, long_window=20):
    signals = data.copy()
    signals["short_mavg"] = signals["Close"].rolling(short_window).mean()
    signals["long_mavg"] = signals["Close"].rolling(long_window).mean()
    signals["signal"] = np.where(signals["short_mavg"] > signals["long_mavg"], 1, 0)  # Buy if short MA > long MA
    signals["position"] = signals["signal"].diff()  # 1 = Buy, -1 = Sell
    return signals
```

```python
# Walk-Forward Validation (Backtesting)
def walk_forward_validation(data, short_window=5, long_window=20, min_train_size=30):
    predictions, actual_movements = [], []
    for i in range(min_train_size, len(data)):
        train_data = data.iloc[:i]
        prediction = moving_average_strategy(train_data)["signal"].iloc[-1]
        actual = 1 if data["Close"].iloc[i+1] > data["Close"].iloc[i] else 0
        predictions.append(prediction)
        actual_movements.append(actual)
    return accuracy_score(actual_movements, predictions)  # Evaluate strategy
```

---

### **3. Data Scaling & Normalization**
- **MinMaxScaler** (normalizes data between `[0, 1]`)
- **RobustScaler** (resistant to outliers)

#### **Code Snippet:**
```python
# Normalize data
if USE_ROBUST_SCALER:
    scaler = RobustScaler()
else:
    scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
```

---

### **4. Visualization Techniques**
- **Prediction vs. Actual Plots**
- **Error Distribution (Violin Plots)**
- **Feature Importance Analysis**
- **Model Training Performance (Loss & MAE curves)**

#### **Code Snippets:**
```python
# Plotting Predictions vs Actual
plt.plot(dates, y_true, label="Actual", color="blue")
plt.plot(dates, y_pred, label="Predicted", color="red", linestyle="--")
plt.fill_between(dates, y_true, y_pred, color="gray", alpha=0.3, label="Error")
```

```python
# Error Distribution (Violin Plot)
plt.violinplot(all_errors, showmeans=True)
plt.xticks(range(1, len(stock_names)+1), stock_names, rotation=45)
```

---

### **5. Model Optimization Techniques**
- **Early Stopping** (prevents overfitting)
- **Learning Rate Reduction on Plateau**
- **Dropout Layers** (for regularization)

#### **Code Snippet:**
```python
# Callbacks for Training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
model.fit(..., callbacks=[early_stopping, reduce_lr])
```

---

### **Summary of Techniques Used:**
| **Category**               | **Technique**                          | **Purpose** |
|----------------------------|----------------------------------------|-------------|
| **Deep Learning**          | LSTM/GRU with Attention                | Time-series forecasting |
| **Technical Analysis**     | Moving Average Crossover               | Generate BUY/SELL signals |
| **Feature Engineering**    | RSI, MACD, Bollinger Bands             | Improve model accuracy |
| **Data Preprocessing**     | MinMaxScaler / RobustScaler            | Normalize data |
| **Model Optimization**     | Early Stopping, Dropout, LR Scheduling | Prevent overfitting |
| **Backtesting**            | Walk-Forward Validation                | Evaluate strategy performance |
