# Financial Forecasting App

Basic Financial App using Streamlit and Yahoo API.
Based on Gencay I. medium article:
https://levelup.gitconnected.com/gpt-4o-with-canvas-lets-build-a-financial-streamlit-app-188183a80292

This Streamlit app predicts stock prices using an LSTM neural network. It fetches historical stock data from the Yahoo Finance API and provides predictions based on the data retrieved.

## Features

- **Adjustable Parameters**:
  - Select a stock ticker symbol.
  - Use date pickers for specifying the historical data range.
  - Configure ML model parameters such as training epochs, LSTM layers, and batch size.

- **Data Visualization**:
  - Display clear line graphs of historical stock data and overlay the predicted prices on the same chart using Plotly.

- **Result Download**:
  - Include a "Download as CSV" feature for historical and predicted data with timestamps, actual prices, and predicted prices.

- **Performance Optimization**:
  - Implement Streamlit caching to store API responses temporarily, minimizing repetitive calls and enhancing app speed.

## How to Use

1. **Select Stock Ticker**:
   - Enter the stock ticker symbol in the sidebar.

2. **Specify Date Range**:
   - Use the date pickers to select the start and end dates for the historical data.

3. **Configure Model Parameters**:
   - Adjust the training epochs, LSTM layers, and batch size using the sliders in the sidebar.

4. **View Predictions**:
   - The app will display a line graph of historical stock data with predicted prices overlaid.

5. **Download Data**:
   - Click the "Download as CSV" button to download the historical and predicted data.

## Deployment

1. **Create a GitHub Repository**:
   - Create a new repository on GitHub and push the `financial_forecasting_app.py` file to this repository.

2. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud).
   - Log in with your GitHub account.
   - Click on "New app" and select your GitHub repository.
   - Follow the instructions to deploy your app.

## Requirements

- Python 3.7+
- Streamlit
- Yahoo Finance API
- Plotly
- Pandas
- Scikit-learn
- TensorFlow

## Installation

```bash
pip install streamlit yfinance plotly pandas scikit-learn tensorflow
