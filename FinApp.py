import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from io import BytesIO

# Streamlit App Title
st.title("Stock Price Prediction App")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# Stock ticker input
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", value="AAPL")

# Date range inputs
date_range = st.sidebar.date_input(
    "Select Historical Data Range:", [pd.to_datetime('2022-01-01'), pd.to_datetime('2023-01-01')]
)

# ML model parameters
epochs = st.sidebar.number_input("Training Epochs:", min_value=1, max_value=100, value=10)
lstm_units = st.sidebar.number_input("LSTM Layers:", min_value=1, max_value=256, value=50)
batch_size = st.sidebar.number_input("Batch Size:", min_value=1, max_value=512, value=32)

# Function to fetch stock data
@st.cache
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch historical stock data
if date_range and len(date_range) == 2:
    start_date, end_date = date_range
    stock_data = fetch_stock_data(ticker, start_date, end_date)
else:
    stock_data = None

if stock_data is not None and not stock_data.empty:
    st.write(f"Displaying data for {ticker} from {start_date} to {end_date}")
    st.dataframe(stock_data)

    # Normalize data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    # Prepare training and testing datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape data for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    with st.spinner('Training the model...'):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    st.success('Model trained successfully!')

    # Predict on test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Prepare actual vs. predicted data
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    predicted_prices = predictions.flatten()
    timestamps = stock_data.index[-len(predicted_prices):]

    result_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Actual Prices': actual_prices.flatten(),
        'Predicted Prices': predicted_prices
    })

    # Plot data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df['Timestamp'], y=result_df['Actual Prices'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=result_df['Timestamp'], y=result_df['Predicted Prices'], mode='lines', name='Predicted Prices'))

    st.plotly_chart(fig)

    # Download results
    csv_data = result_df.to_csv(index=False).encode()
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_data,
        file_name=f"{ticker}_predictions.csv",
        mime='text/csv'
    )

    # How it works section
    st.markdown("""## How it Works
    1. Enter the stock ticker symbol (e.g., AAPL for Apple Inc.).
    2. Select the date range for historical data.
    3. Adjust the machine learning model parameters as needed.
    4. View the historical data, predicted prices, and download results in CSV format.
    """)

else:
    st.error("No data available. Please check the stock ticker or date range.")
