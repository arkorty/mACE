import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Directory containing stock data files
DATA_DIR = "stockdata/"

# Define training sequence length
N = 60


def create_sequences(data, N):
    X, y = [], []
    for i in range(N, len(data)):
        X.append(data[i - N : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model():
    inputs = Input(shape=(N, 1))
    x = LSTM(units=100, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(units=100, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(units=50, activation="relu")(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Initialize lists for overall accuracy metrics
all_mae, all_rmse = [], []

runs = 0

# Process each stock file
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv") and runs < 20:
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path, parse_dates=["Datetime"], index_col="Datetime")

        # Select only 'Close' price for prediction
        data = df[["Close"]].values

        # Skip files with insufficient data
        if len(data) < N:
            print(f"Skipping {file} (not enough data)")
            continue
        else:
            runs += 1

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Create sequences
        X, y = create_sequences(data_scaled, N)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split into train and test sets using a rolling window approach
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train a separate model for each stock
        model = build_lstm_model()
        early_stopping = EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping],
        )

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        all_mae.append(mae)
        all_rmse.append(rmse)

        print(f"Stock: {file}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Compute overall accuracy
if all_mae and all_rmse:
    avg_mae = np.mean(all_mae)
    avg_rmse = np.mean(all_rmse)
    print(f"Overall MAE: {avg_mae:.4f}, Overall RMSE: {avg_rmse:.4f}")
else:
    print("No valid stock data files found for training.")
