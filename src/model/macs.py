import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function to load and prepare the stock data
def load_stock_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Convert datetime to proper format
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Set Datetime as index
    df.set_index("Datetime", inplace=True)

    # Sort by date
    df.sort_index(inplace=True)

    return df


# Function to implement moving average crossover strategy
def moving_average_strategy(data, short_window=5, long_window=20):
    # Make a copy of the data
    signals = data.copy()

    # Create short and long moving averages
    signals["short_mavg"] = (
        signals["Close"].rolling(window=short_window, min_periods=1).mean()
    )
    signals["long_mavg"] = (
        signals["Close"].rolling(window=long_window, min_periods=1).mean()
    )

    # Create signals: 1 for buy, -1 for sell, 0 for hold
    signals["signal"] = 0
    signals["signal"] = np.where(signals["short_mavg"] > signals["long_mavg"], 1, 0)

    # Generate trading orders
    signals["position"] = signals["signal"].diff()

    return signals


# Function to calculate strategy performance
def calculate_performance(signals):
    # Initial capital
    initial_capital = 100000

    # Create a DataFrame with positions
    positions = pd.DataFrame(index=signals.index)
    positions["price"] = signals["Close"]
    positions["signal"] = signals["signal"]

    # Buy shares
    positions["shares"] = positions["signal"] * 100  # Buy 100 shares when signal is 1

    # Calculate investment value
    positions["cash"] = (
        initial_capital - (positions["shares"] * positions["price"]).cumsum()
    )
    positions["holdings"] = positions["shares"] * positions["price"]
    positions["total"] = positions["cash"] + positions["holdings"]

    # Calculate returns
    positions["returns"] = positions["total"].pct_change()

    # Check if there's data in the positions DataFrame
    if len(positions) == 0:
        return positions, 0, 0

    # Calculate metrics - using .iloc[-1] instead of [-1] to avoid deprecation warning
    final_value = (
        positions["total"].iloc[-1] if not positions["total"].empty else initial_capital
    )
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Calculate Sharpe ratio if there are returns
    if not positions["returns"].empty and positions["returns"].notna().any():
        sharpe_ratio = (
            positions["returns"].mean() / positions["returns"].std() * np.sqrt(252)
            if positions["returns"].std() != 0
            else 0
        )
    else:
        sharpe_ratio = 0

    return positions, total_return, sharpe_ratio


# Function to perform walk-forward validation
def walk_forward_validation(data, short_window=5, long_window=20, min_train_size=30):
    if len(data) <= min_train_size:
        return None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    # Initialize results storage
    predictions = []
    actual_movements = []

    # For each test point after minimum training size
    for i in range(min_train_size, len(data)):
        # Split data into train and test
        train_data = data.iloc[:i]
        test_point = data.iloc[i : i + 1]

        if train_data.empty or test_point.empty:
            continue

        # Apply strategy to training data
        train_signals = moving_average_strategy(train_data, short_window, long_window)

        # Get the last signal from training data
        if train_signals.empty:
            continue

        prediction = train_signals["signal"].iloc[-1]

        # Determine actual movement
        if i + 1 < len(data):
            actual = 1 if data["Close"].iloc[i + 1] > data["Close"].iloc[i] else 0
        else:
            # For the last point, we can't determine future movement
            continue

        predictions.append(prediction)
        actual_movements.append(actual)

    # Calculate metrics if we have predictions
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


# Function to visualize prediction accuracy
def plot_accuracy_metrics(all_metrics, title="Prediction Accuracy Across Stocks"):
    stocks = list(all_metrics.keys())

    if not stocks:
        print("No metrics to plot")
        return None

    metrics = ["accuracy", "precision", "recall", "f1"]

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set width of bars
    bar_width = 0.2
    index = np.arange(len(stocks))

    # Plot bars
    for i, metric in enumerate(metrics):
        values = [all_metrics[stock][metric] for stock in stocks]
        ax.bar(index + i * bar_width, values, bar_width, label=metric.capitalize())

    # Add labels and title
    ax.set_xlabel("Stocks")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(stocks, rotation=45, ha="right")
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add a horizontal line at 0.5 (random guess)
    ax.axhline(y=0.5, color="r", linestyle="-", alpha=0.3, label="Random Guess")

    plt.tight_layout()
    return fig


# Function to plot overall accuracy metrics
def plot_overall_metrics(all_metrics):
    if not all_metrics:
        print("No metrics to plot")
        return None

    # Calculate average metrics
    avg_metrics = {}
    metrics = ["accuracy", "precision", "recall", "f1"]

    for metric in metrics:
        values = [m[metric] for m in all_metrics.values() if m[metric] > 0]
        avg_metrics[metric] = np.mean(values) if values else 0

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bar chart
    bars = ax.bar(
        avg_metrics.keys(),
        avg_metrics.values(),
        color=["blue", "green", "orange", "purple"],
    )

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Add a horizontal line at 0.5 (random guess)
    ax.axhline(y=0.5, color="r", linestyle="-", alpha=0.3, label="Random Guess")

    # Add labels and title
    ax.set_ylabel("Score")
    ax.set_title("Average Prediction Metrics Across All Stocks")
    ax.set_ylim(0, 1)
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


# Function to analyze a stock using walk-forward validation
def analyze_stock_walkforward(
    file_path, short_window=5, long_window=20, min_train_size=30
):
    try:
        # Get stock name from filename
        stock_name = os.path.basename(file_path).split(".")[0]

        # Load data
        data = load_stock_data(file_path)

        # Check if data is not empty and has enough records
        if data.empty or len(data) < min_train_size + 1:
            print(
                f"Insufficient data for {stock_name} (need at least {min_train_size+1} data points)"
            )
            return None, None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        # Perform walk-forward validation
        results_df, metrics = walk_forward_validation(
            data, short_window, long_window, min_train_size
        )

        # Display results
        print(f"\nWalk-Forward Validation for {stock_name}:")
        print(f"Strategy: Moving Average Crossover ({short_window}/{long_window})")
        print(
            f"Number of validation points: {len(results_df) if results_df is not None else 0}"
        )
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        # Apply the strategy to the full dataset for comparison
        signals = moving_average_strategy(data, short_window, long_window)
        positions, total_return, sharpe_ratio = calculate_performance(signals)

        print(
            f"Full dataset - Total Return: {total_return:.2f}%, Sharpe Ratio: {sharpe_ratio:.4f}"
        )

        return results_df, signals, metrics

    except Exception as e:
        print(f"Error analyzing {os.path.basename(file_path)}: {str(e)}")
        return None, None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}


# Main function to run the analysis
def main():
    # Parameters
    short_window = 5  # 5-day moving average
    long_window = 20  # 20-day moving average
    min_train_size = 30  # Minimum data points for training
    max_stocks = 500  # Maximum number of stocks to analyze

    # Find all CSV files in the stockdata directory
    stock_files = glob.glob("stockdata/*.csv")

    if not stock_files:
        print("No stock data files found in 'stockdata/' directory.")
        return

    # Limit to first 40 stocks
    stock_files = stock_files[:max_stocks]

    # Create a results dictionary to store all analysis results
    results = {}
    all_metrics = {}

    # Analyze each stock
    for file_path in stock_files:
        stock_name = os.path.basename(file_path).split(".")[0]

        validation_results, signals, metrics = analyze_stock_walkforward(
            file_path, short_window, long_window, min_train_size
        )

        if validation_results is not None:
            results[stock_name] = validation_results
            all_metrics[stock_name] = metrics

            # Output the number of validation runs for this stock
            if validation_results is not None:
                print(
                    f"Completed {len(validation_results)} validation runs for {stock_name}"
                )

    # Check if we have any valid results
    if not results:
        print("No valid analysis results were generated.")
        return

    os.makedirs("macs", exist_ok=True)

    # Plot individual stock metrics
    accuracy_fig = plot_accuracy_metrics(all_metrics)
    if accuracy_fig:
        accuracy_fig.savefig("macs/stock_accuracy_metrics.png")
        plt.close(accuracy_fig)

    # Plot overall metrics
    overall_fig = plot_overall_metrics(all_metrics)
    if overall_fig:
        overall_fig.savefig("macs/overall_accuracy_metrics.png")
        plt.close(overall_fig)

    # Sort stocks by accuracy
    sorted_stocks = sorted(
        all_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    print("\n=== Top Performing Stocks by Prediction Accuracy ===")
    for i, (stock_name, metrics) in enumerate(sorted_stocks[:10]):
        print(
            f"{i+1}. {stock_name}: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}"
        )

    # Calculate average metrics across all stocks
    avg_accuracy = np.mean([m["accuracy"] for m in all_metrics.values()])
    avg_precision = np.mean([m["precision"] for m in all_metrics.values()])
    avg_recall = np.mean([m["recall"] for m in all_metrics.values()])
    avg_f1 = np.mean([m["f1"] for m in all_metrics.values()])

    print("\n=== Overall Strategy Performance ===")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


def plot_stock_analysis(stock_name, data, signals, results_df, metrics):
    os.makedirs(f"macs/{stock_name}", exist_ok=True)

    # Plot stock price with moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="black")
    plt.plot(
        data.index,
        signals["short_mavg"],
        label="Short MA (5)",
        color="blue",
        linestyle="dashed",
    )
    plt.plot(
        data.index,
        signals["long_mavg"],
        label="Long MA (20)",
        color="red",
        linestyle="dashed",
    )
    plt.legend()
    plt.title(f"{stock_name} - Moving Average Crossover")
    plt.savefig(f"macs/{stock_name}/price_moving_avg.png")
    plt.close()

    # Plot buy/sell signals
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="black", alpha=0.6)
    buy_signals = data.index[signals["position"] == 1]
    sell_signals = data.index[signals["position"] == -1]
    plt.scatter(
        buy_signals,
        data.loc[buy_signals]["Close"],
        label="Buy Signal",
        marker="^",
        color="green",
    )
    plt.scatter(
        sell_signals,
        data.loc[sell_signals]["Close"],
        label="Sell Signal",
        marker="v",
        color="red",
    )
    plt.legend()
    plt.title(f"{stock_name} - Buy/Sell Signals")
    plt.savefig(f"macs/{stock_name}/buy_sell_signals.png")
    plt.close()

    # Plot prediction results
    if results_df is not None and not results_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(
            results_df.index,
            results_df["Predicted"],
            label="Predicted",
            linestyle="dotted",
            color="blue",
        )
        plt.plot(
            results_df.index,
            results_df["Actual"],
            label="Actual",
            linestyle="dotted",
            color="red",
        )
        plt.legend()
        plt.title(f"{stock_name} - Prediction Accuracy")
        plt.savefig(f"macs/{stock_name}/prediction_accuracy.png")
        plt.close()

    # Save metrics to a text file
    with open(f"macs/{stock_name}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")


def analyze_stock_walkforward(
    file_path, short_window=5, long_window=20, min_train_size=30
):
    try:
        stock_name = os.path.basename(file_path).split(".")[0]
        data = load_stock_data(file_path)

        if data.empty or len(data) < min_train_size + 1:
            print(f"Insufficient data for {stock_name}")
            return None, None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        results_df, metrics = walk_forward_validation(
            data, short_window, long_window, min_train_size
        )
        signals = moving_average_strategy(data, short_window, long_window)

        plot_stock_analysis(stock_name, data, signals, results_df, metrics)

        return results_df, signals, metrics
    except Exception as e:
        print(f"Error analyzing {stock_name}: {e}")
        return None, None, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}


if __name__ == "__main__":
    main()
