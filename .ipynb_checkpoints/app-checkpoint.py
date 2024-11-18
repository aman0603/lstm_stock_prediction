import os
from flask import Flask, render_template, request, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

app = Flask(__name__)

# Load the trained LSTM model
model = load_model('C:/Users/amanp/Desktop/lstm_stock/lstm_stock_model.h5')

# Define a function to load and preprocess data for prediction
def load_data(ticker):
    data = yf.download(ticker, "2010-01-01", pd.to_datetime('today'))
    
    # Check if the data is empty
    if data.empty:
        raise ValueError(f"No data fetched for ticker: {ticker}. Please check the ticker symbol.")
    
    data.reset_index(inplace=True)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = data.iloc[:, 4:5].values
    
    # If data is empty, raise an error
    if train_close.size == 0:
        raise ValueError("The 'Close' data is empty, unable to preprocess.")
    
    data_training_array = scaler.fit_transform(train_close)
    x_train = []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100: i])
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, scaler, data

def create_plot(data, predicted_price, ticker, days=100):
    # Get the last 'days' rows from the data (e.g., the last 30 days)
    data_recent = data.tail(days)
    
    # Plot the actual stock prices and the predicted price
    plt.figure(figsize=(10, 6))
    plt.plot(data_recent['Date'], data_recent['Close'], label='Actual Price', color='blue')
    plt.axvline(x=data_recent['Date'].iloc[-1], color='red', linestyle='--', label='Prediction Point')
    
    # Add predicted future price point
    future_date = data_recent['Date'].iloc[-1] + pd.Timedelta(days=1)
    plt.scatter(future_date, predicted_price, color='green', label=f'Predicted Price: ${predicted_price:.2f}')
    
    # Format the plot
    plt.title(f'{ticker} Stock Price and Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()

    # Save the plot
    plot_path = 'static/stock_price_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path


# Define the route for the home page
# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        ticker = request.form['ticker']
        try:
            data = load_data(ticker)
            x_train, scaler, data = preprocess_data(data)
            
            # Predict the next day's price (or adjust for multiple days if necessary)
            prediction = model.predict(x_train[-1:].reshape((1, 100, 1)))
            predicted_price = scaler.inverse_transform(prediction)[0][0]
            
            # Get the current stock price (convert to float)
            current_price = float(data['Close'].iloc[-1])  # Last available close price
            
            # Generate the plot for the last 30 days
            plot_path = create_plot(data, predicted_price, ticker, days=100)
            
            return render_template('home.html', ticker=ticker, predicted_price=predicted_price, 
                                   current_price=current_price, plot_url=plot_path)
        
        except ValueError as e:
            return render_template('home.html', error=str(e), ticker=ticker)
    
    return render_template('home.html')


# Serve the generated plot from static folder
@app.route('/static/<filename>')
def send_image(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)

