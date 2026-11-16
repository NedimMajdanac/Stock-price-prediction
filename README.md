# Stock Price Prediction

An educational stock-price forecasting project using a PyTorch LSTM neural network. The project downloads historical Apple (AAPL) closing prices with [`yfinance`](https://github.com/ranaroussi/yfinance), trains on rolling sequences of historical prices, evaluates the model with root mean squared error (RMSE), and plots actual versus predicted prices.

> **Important:** This project is for experimentation and learning, not financial advice. Historical price patterns do not guarantee future results, and the model does not account for news, fundamentals, market conditions, transaction costs, or uncertainty.

## Features

- Downloads historical market data from Yahoo Finance.
- Uses the closing price as the prediction target.
- Standardizes the price series before training.
- Builds 30-day rolling sequences.
- Splits the sequences into chronological training and test sets.
- Trains a two-layer LSTM with PyTorch.
- Uses CUDA automatically when a compatible GPU is available.
- Converts predictions back to the original price scale.
- Reports training and test RMSE.
- Visualizes actual prices, predicted prices, and prediction error.

## Project structure

```text
.
├── Main.ipynb   # Interactive notebook version of the experiment
├── main.py      # Script version of the experiment
└── README.md    # Project documentation
```

## Requirements

- Python 3.9 or newer is recommended.
- Internet access is required because market data is downloaded at runtime.
- A CUDA-capable GPU is optional; the code falls back to the CPU.

The main dependencies are:

- NumPy
- pandas
- Matplotlib
- yfinance
- PyTorch
- scikit-learn

## Installation

Create and activate a virtual environment, then install the dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib yfinance scikit-learn torch
```

If PowerShell prevents script activation, activate the environment using:

```powershell
.\.venv\Scripts\activate.bat
```

For a CUDA-enabled PyTorch installation, use the command recommended by the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your operating system and CUDA version.

## Usage

### Run the Python script

From the project directory:

```powershell
python main.py
```

The script will:

1. Download AAPL data from 1 January 2020 onward.
2. Create training and test sequences.
3. Train the LSTM for 200 epochs.
4. Print periodic training loss.
5. Print the training and test RMSE values.
6. Display a chart containing predictions and errors.

### Run the notebook

Start Jupyter:

```powershell
python -m pip install notebook
jupyter notebook
```

Open `Main.ipynb` and run the cells from top to bottom. The notebook is useful for inspecting intermediate data, tensors, model output, and plots interactively.

## How the model works

The current implementation uses the following pipeline:

1. **Data retrieval:** `yfinance` downloads historical data for the `AAPL` ticker.
2. **Feature selection:** only the `Close` price is used.
3. **Scaling:** `StandardScaler` transforms prices before model training.
4. **Sequence creation:** each sample contains 30 consecutive observations.
5. **Target selection:** the final observation in each sequence is the target; the preceding observations are used as input.
6. **Data split:** the first 80% of sequences are used for training and the remaining 20% for testing.
7. **Prediction:** a two-layer LSTM with 32 hidden units predicts the next scaled closing price.
8. **Evaluation:** predictions and targets are inverse-transformed to price units, then evaluated with RMSE.

The main hyperparameters are defined directly in `main.py`:

```python
seq_length = 30
hidden_dim = 32
num_layers = 2
learning_rate = 0.01
num_epochs = 200
```

## Output

The script displays:

- Training loss during optimization.
- Training RMSE.
- Test RMSE.
- A comparison of actual and predicted test prices.
- Absolute prediction error over the test period.

Results can vary depending on the data returned by Yahoo Finance, the installed library versions, random initialization, and whether the model runs on a CPU or GPU.

## Limitations and possible improvements

This is a minimal, single-series demonstration. Useful next steps include:

- Adding explicit random seeds for reproducibility.
- Fitting the scaler only on the training period to avoid data leakage.
- Downloading multiple features such as open, high, low, volume, and technical indicators.
- Replacing hard-coded settings with command-line arguments or a configuration file.
- Adding validation data and early stopping.
- Saving and loading trained model checkpoints.
- Comparing the LSTM against naive, linear, tree-based, or transformer baselines.
- Adding automated tests and experiment tracking.
- Handling missing data and Yahoo Finance download failures explicitly.
- Evaluating against a walk-forward or rolling-origin validation strategy.

## License

No license is currently specified for this repository. Add a license before redistributing the project or incorporating it into another application.
