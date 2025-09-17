import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title('Stock Price Prediction App')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Load data
st.info("Loading data...")
df_train = yf.download(user_input, start='2010-01-01', end='2023-12-31')
df_recent = yf.download(user_input, start='2024-01-01', end=datetime.now().strftime('%Y-%m-%d'))

if len(df_train) == 0 or len(df_recent) == 0:
    st.error("No data found for this ticker")
    st.stop()

st.success("Data loaded successfully!")

current = df_recent['Close'].iloc[-1]
st.subheader('Stock Information')
st.write("Current Price: $", round(float(current), 2))

# Chart
st.subheader('Recent Price Chart')
fig = plt.figure(figsize=(10, 6))
plt.plot(df_recent['Close'])
plt.title('Recent Price Trend')
plt.ylabel('Price ($)')
st.pyplot(fig)

# Prediction
st.subheader('Price Prediction')

try:
    model = load_model('keras_model.h5')
    st.success("Model loaded!")
    
    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train['Close'].values.reshape(-1, 1))
    
    if len(df_recent) >= 100:
        # Get last 100 days
        last_100 = df_recent['Close'].iloc[-100:].values
        last_100 = last_100.reshape(-1, 1)
        last_100_scaled = scaler.transform(last_100)
        
        # Prepare for model
        X_test = last_100_scaled.reshape(1, 100, 1)
        
        # Predict
        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        predicted_price = float(pred[0][0])
        
        # Show results
        st.write("Today's Price: $", round(float(current), 2))
        st.write("Tomorrow's Prediction: $", round(predicted_price, 2))
        
        change = predicted_price - float(current)
        st.write("Expected Change: $", round(change, 2))
        
        if change > 0:
            st.success("Price expected to go UP")
        else:
            st.error("Price expected to go DOWN")
        
        # Prediction chart
        fig2 = plt.figure(figsize=(10, 6))
        recent_prices = df_recent['Close'].iloc[-10:].values
        days_recent = list(range(10))  # 0 to 9 for recent days
        
        plt.plot(days_recent, recent_prices, 'b-o', label='Recent Prices')
        plt.plot([10], [predicted_price], 'ro', markersize=10, label='Prediction')
        plt.plot([9, 10], [float(recent_prices[-1]), predicted_price], 'r--')
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.title('Price Prediction')
        plt.legend()
        st.pyplot(fig2)
        
    else:
        st.error("Need at least 100 days of recent data")
        
except FileNotFoundError:
    st.error("Model file not found! Run the notebook first.")
except Exception as e:
    st.error("Error: " + str(e))