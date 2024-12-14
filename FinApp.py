import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title("Stock Price Prediction App")

    # Get user input
    ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if ticker_symbol:
        # Fetch data from Yahoo Finance
        df = yf.download(ticker_symbol, start=start_date, end=end_date)

        # Data preprocessing
        data = df['Close'].values
        data = data.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Split data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        test_size = len(scaled_data) - train_size
        train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

        # Convert an array of values into a dataset matrix
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        # Reshape into X=t and Y=t+1
        look_back = 60
        X_train, y_train = create_dataset(train_data, look_back)
        X_test, y_test = create_dataset(test_data, look_back)

        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=25, batch_size=32)

        # Make predictions
        train_Predict = model.predict(X_train)
        test_Predict = model.predict(X_test)

        # Invert predictions
        train_Predict = scaler.inverse_transform(train_Predict)
        y_train = scaler.inverse_transform([y_train])
        test_Predict = scaler.inverse_transform(test_Predict)
        y_test = scaler.inverse_transform([y_test])

        # Visualize the results
        plt.figure(figsize=(16, 6))
        plt.plot(y_train.flatten(), label='Actual Train Price')
        plt.plot(train_Predict.flatten(), label='Predicted Train Price')
        plt.title('Stock Price Prediction (Train Data)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        plt.figure(figsize=(16, 6))
        plt.plot(y_test.flatten(), label='Actual Test Price')
        plt.plot(test_Predict.flatten(), label='Predicted Test Price')
        plt.title('Stock Price Prediction (Test Data)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # Predict future prices
        x_input = test_data[len(test_data)-look_back:].reshape(1, -1)
        x_input = x_input.reshape((1, x_input.shape[1], 1))
        future_prediction = model.predict(x_input)
        future_prediction = scaler.inverse_transform(future_prediction)

        st.write(f"Predicted price for the next day: {future_prediction[0][0]}")

        # Download data as CSV
        df_results = pd.DataFrame({'Date': df.index, 'Actual Price': df['Close'], 'Predicted Price': np.concatenate((train_Predict.flatten(), test_Predict.flatten(), future_prediction.flatten()))})
        csv = df_results.to_csv(index=False)
        st.download_button("Download Results", data=csv, file_name="stock_predictions.csv", mime='text/csv')

if __name__ == '__main__':
    main()
