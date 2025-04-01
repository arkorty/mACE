import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Bidirectional,
    GRU,
    Layer,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from datetime import datetime
import seaborn as sns

# Directory containing stock data files
DATA_DIR = "stockdata/"

# Define training sequence length
SEQUENCE_LENGTH = 60

# Experiment settings
USE_ATTENTION = True
USE_BIDIRECTIONAL = True
MODEL_TYPE = "LSTM"  # Options: "LSTM", "GRU"
SHARED_MODEL = False  # Set to True to train a single model on all stocks
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 100

# Feature engineering settings
USE_TECHNICAL_INDICATORS = True
USE_ROBUST_SCALER = True


# Custom attention layer as a proper Keras Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        self.b = self.add_weight(
            name="attention_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Shape: (batch_size, seq_len, 1)
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)

        # Remove the last dimension. Shape: (batch_size, seq_len)
        e = tf.keras.backend.squeeze(e, axis=-1)

        # Compute the weights. Shape: (batch_size, seq_len)
        alpha = tf.keras.backend.softmax(e)

        # Reshape alpha to match the input shape. Shape: (batch_size, seq_len, 1)
        alpha_expanded = tf.keras.backend.expand_dims(alpha, axis=-1)

        # Compute the context vector. Shape: (batch_size, features)
        context = tf.keras.backend.sum(x * alpha_expanded, axis=1)

        return context


def add_technical_indicators(df):
    """Add technical indicators as additional features."""
    # Copy the dataframe to avoid modifying the original
    df = df.copy()

    # Simple Moving Averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Exponential Moving Averages
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_middle"] = df["Close"].rolling(window=20).mean()
    df["BB_std"] = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_middle"] - 2 * df["BB_std"]

    # MACD
    df["MACD"] = (
        df["Close"].ewm(span=12, adjust=False).mean()
        - df["Close"].ewm(span=26, adjust=False).mean()
    )
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Price rate of change
    df["ROC"] = df["Close"].pct_change(periods=5) * 100

    # Fill NaN values with forward fill, then backward fill
    df = df.ffill().bfill()  # Fixed the deprecated method

    return df


def create_sequences(data, seq_length):
    """Create sequences for training with multiple features."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, 0])  # Still predicting only the closing price
    return np.array(X), np.array(y)


def build_advanced_model(
    input_shape, model_type="LSTM", use_attention=False, use_bidirectional=False
):
    """Build an enhanced model with various architecture options."""
    inputs = Input(shape=input_shape)

    # Choose layer type
    layer_class = LSTM if model_type == "LSTM" else GRU

    # First recurrent layer
    if use_bidirectional:
        x = Bidirectional(layer_class(units=128, return_sequences=True))(inputs)
    else:
        x = layer_class(units=128, return_sequences=True)(inputs)

    x = Dropout(DROPOUT_RATE)(x)

    # Second recurrent layer
    if use_bidirectional:
        x = Bidirectional(layer_class(units=64, return_sequences=use_attention))(x)
    else:
        x = layer_class(units=64, return_sequences=use_attention)(x)

    x = Dropout(DROPOUT_RATE)(x)

    # Add attention mechanism if specified
    if use_attention:
        x = AttentionLayer()(x)

    # Dense layers
    x = Dense(units=32, activation="relu")(x)
    outputs = Dense(units=1)(x)

    model = Model(inputs, outputs)

    # Use Adam optimizer with specified learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

    return model


def visualize_predictions(stock_name, y_true, y_pred, dates=None):
    """Visualize predictions vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.title(f"Stock Price Prediction for {stock_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Use dates if provided, otherwise use indices
    x_values = dates if dates is not None else range(len(y_true))

    plt.plot(x_values, y_true, label="Actual", color="blue")
    plt.plot(x_values, y_pred, label="Predicted", color="red", linestyle="--")

    # Calculate error metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.text(
        0.02,
        0.95,
        f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add a visual indication of prediction error
    plt.fill_between(
        x_values,
        y_true.flatten(),
        y_pred.flatten(),
        color="gray",
        alpha=0.3,
        label="Error",
    )

    plt.tight_layout()

    # Save the visualization
    os.makedirs("predictions", exist_ok=True)
    plt.savefig(f"predictions/{stock_name}_prediction.png")
    plt.close()


def visualize_error_distribution(all_errors, stock_names):
    """Visualize the distribution of prediction errors across stocks."""
    plt.figure(figsize=(12, 6))
    plt.title("Prediction Error Distribution Across Stocks")

    # Ensure we have at least one stock
    if len(all_errors) > 0:
        plt.violinplot(all_errors, showmeans=True)
        plt.xticks(range(1, len(stock_names) + 1), stock_names, rotation=45)
        plt.ylabel("Absolute Error")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(
            0.5,
            0.5,
            "No data available for visualization",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.tight_layout()

    # Save the visualization
    os.makedirs("analysis", exist_ok=True)
    plt.savefig("analysis/error_distribution.png")
    plt.close()


def visualize_feature_importance(model, feature_names):
    """Visualize the importance of different features."""
    try:
        # Simple feature importance based on the weights of the first dense layer
        weights = model.layers[-2].get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.barh(range(len(feature_names)), importance)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Importance Score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the visualization
        os.makedirs("analysis", exist_ok=True)
        plt.savefig("analysis/feature_importance.png")
        plt.close()
    except Exception as e:
        print(f"Could not visualize feature importance: {str(e)}")


def visualize_model_performance(history):
    """Visualize model training performance."""
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"])
        plt.legend(["Train", "Validation"], loc="upper right")
    else:
        plt.legend(["Train"], loc="upper right")

    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)

    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"])

    if "val_mae" in history.history:
        plt.plot(history.history["val_mae"])
        plt.legend(["Train", "Validation"], loc="upper right")
    else:
        plt.legend(["Train"], loc="upper right")

    plt.title("Model MAE")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the visualization
    os.makedirs("analysis", exist_ok=True)
    plt.savefig("analysis/model_performance.png")
    plt.close()


def main():
    # Create directories for outputs
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize lists for overall accuracy metrics
    all_mae, all_rmse = [], []
    all_errors = []
    stock_names = []

    # Lists to store all training data if using a shared model
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    all_scalers = []

    # Process each stock file
    valid_files = 0
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv") and valid_files < 20:
            stock_name = file.split(".")[0]
            file_path = os.path.join(DATA_DIR, file)

            print(f"Processing {stock_name}...")

            try:
                # Load and preprocess data
                df = pd.read_csv(
                    file_path, parse_dates=["Datetime"], index_col="Datetime"
                )

                # Skip files with insufficient data
                if (
                    len(df) < SEQUENCE_LENGTH + 30
                ):  # Ensure we have enough data for training and testing
                    print(f"Skipping {file} (not enough data)")
                    continue

                # Feature engineering
                if (
                    USE_TECHNICAL_INDICATORS
                    and "Open" in df.columns
                    and "High" in df.columns
                    and "Low" in df.columns
                ):
                    df = add_technical_indicators(df)

                    # Select features for prediction - adapt based on your CSV structure
                    available_columns = set(df.columns)
                    desired_features = [
                        "Close",
                        "SMA_5",
                        "SMA_20",
                        "RSI",
                        "MACD",
                        "ROC",
                    ]

                    # Filter features to only include those available
                    features = [f for f in desired_features if f in available_columns]
                else:
                    features = ["Close"]

                # Ensure at least Close price is available
                if "Close" not in df.columns:
                    print(f"Skipping {file} (no Close price column)")
                    continue

                data = df[features].values

                # Skip files with NaN values
                if np.isnan(data).any():
                    print(f"Skipping {file} (contains NaN values)")
                    continue

                valid_files += 1
                stock_names.append(stock_name)

                # Normalize the data
                if USE_ROBUST_SCALER:
                    scaler = RobustScaler()
                else:
                    scaler = MinMaxScaler(feature_range=(0, 1))

                data_scaled = scaler.fit_transform(data)

                # Create sequences
                X, y = create_sequences(data_scaled, SEQUENCE_LENGTH)

                # Ensure we have enough data for training after sequence creation
                if len(X) < 10:  # Arbitrary minimum size
                    print(f"Skipping {file} (not enough sequences created)")
                    continue

                # Split into train, validation, and test sets
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)

                # Adjust sizes if data is limited
                if train_size < 5:
                    train_size = len(X) - 2
                    val_size = 1

                X_train = X[:train_size]
                y_train = y[:train_size]

                X_val = (
                    X[train_size : train_size + val_size] if val_size > 0 else X[:1]
                )  # Use at least one sample for validation
                y_val = y[train_size : train_size + val_size] if val_size > 0 else y[:1]

                X_test = X[train_size + val_size :]
                y_test = y[train_size + val_size :]

                # If test set is empty, use the last validation sample
                if len(X_test) == 0:
                    X_test = X_val[-1:]
                    y_test = y_val[-1:]

                # Get dates for test set visualization
                test_dates = df.index[
                    SEQUENCE_LENGTH + train_size + val_size :
                ].tolist()
                if not test_dates and len(df.index) > SEQUENCE_LENGTH:
                    test_dates = [df.index[-1]]

                if SHARED_MODEL:
                    # Store data for shared model
                    all_X_train.append(X_train)
                    all_y_train.append(y_train)
                    all_X_test.append(X_test)
                    all_y_test.append(y_test)
                    all_scalers.append((scaler, features))
                else:
                    # Train a separate model for each stock
                    print(f"Training model for {stock_name}...")

                    # Build model
                    model = build_advanced_model(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        model_type=MODEL_TYPE,
                        use_attention=USE_ATTENTION,
                        use_bidirectional=USE_BIDIRECTIONAL,
                    )

                    # Callbacks
                    early_stopping = EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    )
                    reduce_lr = ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001
                    )

                    # Check if validation data is available
                    validation_data = (X_val, y_val) if len(X_val) > 0 else None

                    # Train model
                    history = model.fit(
                        X_train,
                        y_train,
                        validation_data=validation_data,
                        epochs=MAX_EPOCHS,
                        batch_size=min(
                            BATCH_SIZE, len(X_train)
                        ),  # Ensure batch size isn't larger than dataset
                        verbose=1,
                        callbacks=[early_stopping, reduce_lr],
                    )

                    # Visualize training performance
                    visualize_model_performance(history)

                    # Save model
                    model.save(f"models/{stock_name}_{timestamp}.h5")

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Transform predictions and test data back to original scale
                    y_pred_inverse = np.zeros((len(y_pred), len(features)))
                    y_pred_inverse[:, 0] = y_pred.flatten()
                    y_pred_inverse = scaler.inverse_transform(y_pred_inverse)[:, 0]

                    y_test_inverse = np.zeros((len(y_test), len(features)))
                    y_test_inverse[:, 0] = y_test
                    y_test_inverse = scaler.inverse_transform(y_test_inverse)[:, 0]

                    # Calculate metrics
                    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
                    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
                    all_mae.append(mae)
                    all_rmse.append(rmse)

                    # Calculate absolute errors for distribution visualization
                    errors = np.abs(y_test_inverse - y_pred_inverse)
                    all_errors.append(errors)

                    print(f"Stock: {stock_name}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

                    # Visualize predictions
                    visualize_predictions(
                        stock_name,
                        y_test_inverse.reshape(-1, 1),
                        y_pred_inverse.reshape(-1, 1),
                        test_dates,
                    )

                    # Visualize feature importance if using multiple features
                    if len(features) > 1:
                        visualize_feature_importance(model, features)

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

    if SHARED_MODEL and all_X_train:
        print("Training shared model on all stocks...")

        try:
            # Combine all training data
            X_train_combined = np.vstack(all_X_train)
            y_train_combined = np.concatenate(all_y_train)

            # Build shared model
            model = build_advanced_model(
                input_shape=(X_train_combined.shape[1], X_train_combined.shape[2]),
                model_type=MODEL_TYPE,
                use_attention=USE_ATTENTION,
                use_bidirectional=USE_BIDIRECTIONAL,
            )

            # Callbacks
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001
            )

            # Create validation set from combined data
            indices = np.arange(len(X_train_combined))
            np.random.shuffle(indices)
            val_size = max(1, int(len(indices) * 0.15))

            val_indices = indices[:val_size]
            train_indices = indices[val_size:]

            X_val_shared = X_train_combined[val_indices]
            y_val_shared = y_train_combined[val_indices]

            X_train_shared = X_train_combined[train_indices]
            y_train_shared = y_train_combined[train_indices]

            # Train shared model
            history = model.fit(
                X_train_shared,
                y_train_shared,
                validation_data=(X_val_shared, y_val_shared),
                epochs=MAX_EPOCHS,
                batch_size=min(BATCH_SIZE, len(X_train_shared)),
                verbose=1,
                callbacks=[early_stopping, reduce_lr],
            )

            # Save shared model
            model.save(f"models/shared_model_{timestamp}.h5")

            # Visualize training performance
            visualize_model_performance(history)

            # Evaluate on each stock's test set
            shared_mae, shared_rmse = [], []
            for i, stock_name in enumerate(stock_names):
                X_test = all_X_test[i]
                y_test = all_y_test[i]
                scaler, features = all_scalers[i]

                # Make predictions
                y_pred = model.predict(X_test)

                # Transform predictions and test data back to original scale
                y_pred_inverse = np.zeros((len(y_pred), len(features)))
                y_pred_inverse[:, 0] = y_pred.flatten()
                y_pred_inverse = scaler.inverse_transform(y_pred_inverse)[:, 0]

                y_test_inverse = np.zeros((len(y_test), len(features)))
                y_test_inverse[:, 0] = y_test
                y_test_inverse = scaler.inverse_transform(y_test_inverse)[:, 0]

                # Calculate metrics
                mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
                rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
                shared_mae.append(mae)
                shared_rmse.append(rmse)

                # Calculate absolute errors for distribution visualization
                errors = np.abs(y_test_inverse - y_pred_inverse)

                print(f"Shared Model on {stock_name}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

                # Visualize predictions
                visualize_predictions(
                    f"{stock_name}_shared",
                    y_test_inverse.reshape(-1, 1),
                    y_pred_inverse.reshape(-1, 1),
                )

            # Add shared model results to overall metrics
            if shared_mae and shared_rmse:
                avg_shared_mae = np.mean(shared_mae)
                avg_shared_rmse = np.mean(shared_rmse)
                print(
                    f"Shared Model Overall: MAE: {avg_shared_mae:.4f}, RMSE: {avg_shared_rmse:.4f}"
                )

        except Exception as e:
            print(f"Error training shared model: {str(e)}")
            import traceback

            traceback.print_exc()

    # Visualize error distribution across stocks
    if all_errors:
        visualize_error_distribution(all_errors, stock_names)

    # Compute overall accuracy
    if all_mae and all_rmse:
        avg_mae = np.mean(all_mae)
        avg_rmse = np.mean(all_rmse)

        # Create summary visualization
        plt.figure(figsize=(12, 6))
        plt.title("Model Performance Across Stocks")

        if len(stock_names) > 1:
            plt.subplot(1, 2, 1)
            plt.bar(stock_names, all_mae)
            plt.axhline(
                y=avg_mae, color="r", linestyle="--", label=f"Avg: {avg_mae:.4f}"
            )
            plt.xlabel("Stock")
            plt.ylabel("MAE")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.bar(stock_names, all_rmse)
            plt.axhline(
                y=avg_rmse, color="r", linestyle="--", label=f"Avg: {avg_rmse:.4f}"
            )
            plt.xlabel("Stock")
            plt.ylabel("RMSE")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                f"Single Stock Performance:\nMAE: {all_mae[0]:.4f}\nRMSE: {all_rmse[0]:.4f}",
                horizontalalignment="center",
                verticalalignment="center",
            )

        plt.tight_layout()
        plt.savefig("analysis/overall_performance.png")
        plt.close()

        print(f"Overall MAE: {avg_mae:.4f}, Overall RMSE: {avg_rmse:.4f}")
    else:
        print("No valid stock data files found for training.")


if __name__ == "__main__":
    main()
