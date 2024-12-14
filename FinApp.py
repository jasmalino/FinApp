import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Streamlit caching to store API responses temporarily
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def create_lstm_model(input_shape, lstm_layers, dense_layers):
    model = Sequential()
    for i in range(lstm_layers):
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    for _ in range(dense_layers):
        model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def main():
    st.title("Financial Forecasting App")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")

    def user_input_features():
        ticker = st.sidebar.text_input("Stock Ticker", 'AAPL')
        start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
        end_date = st.sidebar.date_input("End Date", datetime.date.today())
        epochs = st.sidebar.slider("Training Epochs", 1, 100, 25)
        lstm_layers = st.sidebar.slider("LSTM Layers", 1, 5, 2)
        dense_layers = st.sidebar.slider("Dense Layers", 1, 5, 1)
        batch_size = st.sidebar.slider("Batch Size", 1, 100, 32)
        return ticker, start_date, end_date, epochs, lstm_layers, dense_layers, batch_size

    ticker, start_date, end_date, epochs, lstm_layers, dense_layers, batch_size = user_input_features()

    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    # Prepare data for LSTM
    time_step = 100
    X, y = prepare_data(scaled_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Create and train the LSTM model
    model = create_lstm_model((time_step, 1), lstm_layers, dense_layers)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform([y_train])
    y_test = scaler.inverse_transform([y_test])

    # Plotting
    st.subheader('Stock Price Prediction')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=stock_data.index[time_step:len(train_predict)+time_step], y=train_predict.flatten(), mode='lines', name='Train Predictions'))
    fig.add_trace(go.Scatter(x=stock_data.index[len(train_predict)+(time_step*2)+1:len(stock_data)-1], y=test_predict.flatten(), mode='lines', name='Test Predictions'))
    st.plotly_chart(fig)

    # Download as CSV
    st.subheader('Download Predicted Data')
    predicted_data = pd.DataFrame({
        'Date': stock_data.index[time_step:len(train_predict)+time_step],
        'Actual Price': stock_data['Close'][time_step:len(train_predict)+time_step],
        'Predicted Price': train_predict.flatten()
    })
    st.download_button(
        label="Download as CSV",
        data=predicted_data.to_csv(index=False),
        file_name='predicted_stock_prices.csv',
        mime='text/csv',
    )

    # How it works section
    st.subheader('How it Works')
    st.write("""
    This app uses an LSTM (Long Short-Term Memory) neural network to predict stock prices. The model is trained on historical stock data fetched from the Yahoo Finance API.
    Users can select a stock ticker, specify the historical data range, and configure model parameters such as training epochs, LSTM layers, and batch size.
    The app displays historical stock data and overlays the predicted prices on the same chart using Plotly for interactive and zoomable graphs.
    Users can also download the historical and predicted data as a CSV file.
    """)

if __name__ == '__main__':
    main()
