```markdown
# 📈 mACE - Advanced Stock Analysis and Prediction

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/arkorty/mACE)
[![Version](https://img.shields.io/github/v/release/arkorty/mACE)](https://github.com/arkorty/mACE/releases)
[![License](https://img.shields.io/github/license/arkorty/mACE)](https://github.com/arkorty/mACE/blob/main/LICENSE)
[![Language](https://img.shields.io/github/languages/top/arkorty/mACE)](https://github.com/arkorty/mACE)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)](https://github.com/arkorty/mACE/network/dependencies)

A Python-based toolkit for advanced stock market analysis and prediction, leveraging machine learning models and technical indicators.

<p align="center">
  <img src="https://raw.githubusercontent.com/arkorty/mACE/main/images/mACE_logo.png" alt="mACE Logo" width="400">
</p>

## ✨ Key Features

- **Deep Learning Models:** LSTM and GRU networks for stock price prediction, with optional attention mechanisms.
- **Technical Indicators:** Calculation of SMA, EMA, RSI, MACD, Bollinger Bands, and other key indicators.
- **Moving Average Crossover Strategy:** Analysis of buy/sell signals using backtesting and walk-forward validation.
- **Data Visualization:** Tools for generating plots of predictions, feature importance, and strategy performance.
- **Customizable Parameters:** Easily configurable model and strategy settings to adapt to different markets and assets.

## ⚙️ Prerequisites

- Python 3.7+
- Pip package manager

## 📦 Installation

1. Clone the repository:


```bash
   git clone https://github.com/arkorty/mACE.git
   cd mACE
   

```

2. Install dependencies:

   



```bash
pip install -r requirements.txt
```

## 📂 Directory Structure



```
mACE/
├── src/
│   └── model/
│       ├── lstm.py          # Basic LSTM model for stock prediction
│       ├── lstm2.py         # Advanced LSTM/GRU model with attention
│       ├── macs.py          # Moving Average Crossover Strategy
│       ├── macs2.py         # Enhanced MACS with walk-forward validation
│       └── requirements.txt # Required Python packages
├── stockdata/           # Directory for stock CSV files
├── predictions/         # Generated prediction plots
├── analysis/            # Performance analysis outputs
├── images/               # Directory for images (like the logo)
└── README.md            # Project documentation
```

## 🚀 Usage

### 1. Data Preparation

Ensure your stock data is in CSV format and placed in the 

`stockdata/` directory. The CSV files should have the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

### 2. Run Moving Average Crossover Strategy

```bash
python src/model/macs2.py


```

This script will:

- Load stock data.
- Calculate moving averages (e.g., 50-day and 200-day).
- Generate buy/sell signals based on crossover events.
- Perform walk-forward validation to assess strategy performance.
- Save performance metrics to the `analysis/` directory.

```python
# Example Snippet from macs2.py

import pandas as pd

def moving_average_crossover(data, short_window, long_window):
    """
    Calculates moving averages and generates buy/sell signals.
    """
    short_ma = data['Close'].rolling(window=short_window).mean()
    long_ma = data['Close'].rolling(window=long_window).mean()

    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['signal'][short_ma > long_ma] = 1.0
    signals['positions'] = signals['signal'].diff()
    return signals


```

### 3. Train and Evaluate LSTM Models

- For basic LSTM:

  

```bash
python src/model/lstm.py


```

  This script will:

  - Load stock data.
  - Preprocess data (e.g., scaling).
  - Build and train a basic LSTM model.
  - Evaluate the model's performance.
  - Generate prediction plots in the `predictions/` directory.

```python
# Example Snippet from lstm.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    """
    Builds a basic LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


```

- For advanced LSTM/GRU with attention:

  

```bash
python src/model/lstm2.py


```

  This script will:

  - Load stock data.
  - Preprocess data.
  - Build and train an advanced LSTM/GRU model with attention mechanisms.
  - Evaluate the model's performance.
  - Generate prediction plots and feature importance analysis in the respective directories.

```python
# Example Snippet from lstm2.py

import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    """
    Attention mechanism layer.
    """
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features):
        score = tf.nn.tanh(self.W1(features) + self.W2(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


```

## 📊 Example Outputs

Example outputs (plots and metrics) are automatically saved to the `predictions/` and `analysis/` directories upon script execution. These outputs provide insights into model predictions, feature importance, and strategy performance.

*Ensure these directories are populated with relevant files after running the scripts.*

## 🛠️ Customization

You can customize various parameters in the scripts to tailor the analysis to your specific needs:

- **Sequence length, batch size, and epochs:**  Adjust training parameters in the LSTM scripts.
- **Technical indicators and scaling methods:** Modify the technical indicators used in the analysis and the scaling techniques applied to the data.
- **Moving average windows for MACS:**  Experiment with different moving average windows for the Moving Average Crossover Strategy.

## 🧪 Testing

To run the tests, execute the following command:
```bash
pytest


```
*Note: This project may require a `tests/` folder and `pytest` setup. Ensure appropriate tests are implemented to validate the functionalities of the scripts.*

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.  Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write clear, concise code with comments.
4.  Submit a pull request with a detailed description of your changes.

## 📜 License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.

## 🙏 Acknowledgments

- The TensorFlow and Keras libraries for providing the deep learning framework.
- The Pandas and NumPy libraries for data manipulation and analysis.
- The Matplotlib and Seaborn libraries for data visualization.
```