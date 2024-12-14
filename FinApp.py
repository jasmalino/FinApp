import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit app configuration
st.title("Finance Price Prediction App")
st.write("This app predicts stock prices for the next week using Machine Learning models.")

# Sidebar for user input
st.sidebar.header("Input Settings")
selected_stock = st.sidebar.text_input("Enter the stock ticker symbol (e.g., AAPL):", "AAPL")

# Fetch data from Yahoo Finance
def fetch_data(ticker):
    try:
        data = yf.download(ticker, start="2010-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Preprocess data for LSTM
@st.cache
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X_train = []
    y_train = []
    for i in range(60, len(scaled_data) - 7):  # Use past 60 days to predict the next 7 days
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i:i+7, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

# Build LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=7))  # Predict 7 days

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main workflow
if st.sidebar.button("Predict Prices"):
    # Fetch data
    st.write(f"Fetching data for {selected_stock}...")
    stock_data = fetch_data(selected_stock)

    if stock_data is not None and not stock_data.empty:
        st.write("Data fetched successfully. Displaying the latest data:")
        st.dataframe(stock_data.tail())

        # Preprocess the data
        st.write("Preprocessing the data...")
        X_train, y_train, scaler = preprocess_data(stock_data)

        # Build and train the model
        st.write("Building and training the model...")
        model = build_model()
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

        # Predict future prices
        last_60_days = stock_data['Close'].values[-60:]
        last_60_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_test = np.array([last_60_scaled]).reshape(1, 60, 1)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions).flatten()

        st.write("Predicted prices for the next week:")
        st.write(predictions)

        # Display results
        future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=7)
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
        st.line_chart(prediction_df.set_index("Date"))
    else:
        st.error("Failed to fetch data. Please check the stock ticker symbol.")
