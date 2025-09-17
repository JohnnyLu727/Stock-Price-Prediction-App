# Stock Price Prediction System

A deep learning application that predicts next-day stock prices using **LSTM neural networks** with an interactive web interface.

---

## Features

- **Real-time Data:** Fetches live stock prices from Yahoo Finance API  
- **Deep Learning Model:** Multi-layer LSTM with dropout regularization  
- **Web Dashboard:** Interactive Streamlit application with charts  
- **Price Forecasting:** Next-day stock price predictions  
- **Historical Analysis:** 13+ years of training data (2010–2023)  

---

## Technologies

- **TensorFlow/Keras**  
- **Streamlit**  
- **Pandas & NumPy**  
- **Yahoo Finance API**  
- **Matplotlib**  

---

## Model Architecture

The model is a 4-layer LSTM neural network with progressive units (`50 → 60 → 80 → 120`), dropout regularization ranging from `0.2 → 0.5`, and a dense output layer for next-day price prediction.