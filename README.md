# model;ACE - Advanced Stock Analysis and Prediction

Welcome to **model;ACE**, a comprehensive toolkit for stock market analysis and prediction. This project leverages machine learning models and technical analysis strategies to provide insights into stock price movements.

---

## Features

- **Deep Learning Models**: Predict stock prices using LSTM and GRU models with optional attention mechanisms.
- **Technical Indicators**: Incorporates SMA, EMA, RSI, MACD, Bollinger Bands, and more.
- **Moving Average Crossover Strategy**: Analyze buy/sell signals with walk-forward validation.
- **Visualization Tools**: Generate plots for predictions, feature importance, and strategy performance.
- **Customizable Parameters**: Easily tweak model and strategy settings to suit your needs.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arkorty/mACE.git
   cd mACE
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Structure

```
mACE/
├── src/model/
│   ├── lstm.py          # Basic LSTM model for stock prediction
│   ├── lstm2.py         # Advanced LSTM/GRU model with attention
│   ├── macs.py          # Moving Average Crossover Strategy
│   ├── macs2.py         # Enhanced MACS with walk-forward validation
│   ├── requirements.txt # Required Python packages
├── stockdata/           # Directory for stock CSV files
├── predictions/         # Generated prediction plots
├── analysis/            # Performance analysis outputs
└── README.md            # Project documentation
```

---

## Usage

### 1. Run Moving Average Crossover Strategy
```bash
python src/model/macs2.py
```

### 2. Train and Evaluate LSTM Models
- For basic LSTM:
  ```bash
  python src/model/lstm.py
  ```
- For advanced LSTM/GRU with attention:
  ```bash
  python src/model/lstm2.py
  ```

---

## Example Outputs

### Prediction Visualization
*(Ensure the `predictions/` directory contains relevant plots before referencing them.)*

### Feature Importance
*(Ensure the `analysis/` directory contains feature importance plots before referencing them.)*

### Strategy Performance
*(Ensure the `analysis/` directory contains performance metrics plots before referencing them.)*

---

## Customization

Modify parameters in the respective scripts to customize:
- Sequence length, batch size, and epochs.
- Technical indicators and scaling methods.
- Moving average windows for MACS.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

---

## License

This project is licensed under the [MIT License](LICENSE).
