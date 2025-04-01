import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function to load and prepare the stock data
def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


# Function to implement moving average crossover strategy
def moving_average_strategy(data, short_window=5, long_window=20):
    signals = data.copy()
    signals["short_mavg"] = (
        signals["Close"].rolling(window=short_window, min_periods=1).mean()
    )
    signals["long_mavg"] = (
        signals["Close"].rolling(window=long_window, min_periods=1).mean()
    )
    signals["signal"] = np.where(signals["short_mavg"] > signals["long_mavg"], 1, 0)
    signals["position"] = signals["signal"].diff()
    return signals


# Function to perform walk-forward validation
def walk_forward_validation(data, short_window=5, long_window=20, min_train_size=30):
    if len(data) <= min_train_size:
        return None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    predictions, actual_movements = [], []
    for i in range(min_train_size, len(data)):
        train_data = data.iloc[:i]
        test_point = data.iloc[i : i + 1]
        if train_data.empty or test_point.empty:
            continue
        train_signals = moving_average_strategy(train_data, short_window, long_window)
        if train_signals.empty:
            continue
        prediction = train_signals["signal"].iloc[-1]
        if i + 1 < len(data):
            actual = 1 if data["Close"].iloc[i + 1] > data["Close"].iloc[i] else 0
        else:
            continue
        predictions.append(prediction)
        actual_movements.append(actual)

    metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    if (
        predictions
        and len(predictions) == len(actual_movements)
        and len(set(actual_movements)) > 1
    ):
        try:
            metrics["accuracy"] = accuracy_score(actual_movements, predictions)
            metrics["precision"] = precision_score(
                actual_movements, predictions, zero_division=0
            )
            metrics["recall"] = recall_score(
                actual_movements, predictions, zero_division=0
            )
            metrics["f1"] = f1_score(actual_movements, predictions, zero_division=0)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")

    results_df = pd.DataFrame({"Predicted": predictions, "Actual": actual_movements})
    return results_df, metrics


# Function to visualize buy/sell signals
def plot_signals(data, stock_name, next_day_call):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price", alpha=0.5)
    plt.plot(data.index, data["short_mavg"], label="Short MA", linestyle="--")
    plt.plot(data.index, data["long_mavg"], label="Long MA", linestyle="--")

    buy_signals = data[data["position"] == 1]
    sell_signals = data[data["position"] == -1]

    plt.scatter(
        buy_signals.index,
        buy_signals["Close"],
        marker="^",
        color="g",
        label="Buy Signal",
        alpha=1,
    )
    plt.scatter(
        sell_signals.index,
        sell_signals["Close"],
        marker="v",
        color="r",
        label="Sell Signal",
        alpha=1,
    )

    plt.axvline(
        data.index[-1], color="blue", linestyle="dotted", label="Future Prediction"
    )
    if next_day_call == "BUY":
        plt.scatter(
            data.index[-1],
            data["Close"].iloc[-1],
            color="blue",
            marker="^",
            label="Future Buy Signal",
            s=100,
        )
    else:
        plt.scatter(
            data.index[-1],
            data["Close"].iloc[-1],
            color="purple",
            marker="v",
            label="Future Sell Signal",
            s=100,
        )

    plt.title(f"{stock_name} - Buy/Sell Signals ({next_day_call})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    os.makedirs("predictions", exist_ok=True)
    plt.savefig(f"predictions/{stock_name}.png")
    plt.close()


# Function to analyze a stock using walk-forward validation
def analyze_stock_walkforward(
    file_path, short_window=5, long_window=20, min_train_size=30
):
    try:
        stock_name = os.path.basename(file_path).split(".")[0]
        data = load_stock_data(file_path)
        if data.empty or len(data) < min_train_size + 1:
            return None, None, None, None

        results_df, metrics = walk_forward_validation(
            data, short_window, long_window, min_train_size
        )
        if metrics["accuracy"] < 0.7:
            return None, None, None, None

        signals = moving_average_strategy(data, short_window, long_window)
        next_day_signal = signals["signal"].iloc[-1]
        next_day_call = "BUY" if next_day_signal == 1 else "SELL"

        plot_signals(signals, stock_name, next_day_call)

        return results_df, signals, metrics, next_day_call
    except Exception as e:
        print(f"Error analyzing {os.path.basename(file_path)}: {str(e)}")
        return None, None, None, None


# Main function to run the analysis
def main():
    short_window, long_window, min_train_size, max_stocks = 5, 20, 30, 2000
    stock_files = glob.glob("stockdata/*.csv")[:max_stocks]
    results, all_metrics = {}, {}
    for file_path in stock_files:
        stock_name = os.path.basename(file_path).split(".")[0]
        result = analyze_stock_walkforward(
            file_path, short_window, long_window, min_train_size
        )
        if result[0] is not None:
            validation_results, signals, metrics, next_day_call = result
            results[stock_name] = validation_results
            all_metrics[stock_name] = metrics
            print(f"Stock: {stock_name}")
    if not results:
        print("No valid analysis results were generated.")


if __name__ == "__main__":
    main()
