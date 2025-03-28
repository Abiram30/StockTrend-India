from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import feedparser

app = Flask(__name__)

# Fetch stock data from Yahoo Finance
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    info = stock.info
    return data, info

# Train LSTM model & predict next 60 days
def predict_stock(symbol):
    data, info = get_stock_data(symbol)
    if data.empty:
        return None, None, None, None

    df = data[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(60, len(df_scaled)):
        X.append(df_scaled[i-60:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # Predict next 60 days
    future_prices = []
    last_60_days = df_scaled[-60:]

    for _ in range(60):
        X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        pred_price = model.predict(X_test)
        future_prices.append(pred_price[0][0])
        last_60_days = np.append(last_60_days[1:], pred_price, axis=0)

    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1,1))

    current_price = round(info.get('currentPrice', 0), 2)

    return data, info, future_prices.flatten(), current_price

# Generate interactive stock price graph
def plot_stock(data, future_prices, color):
    past_trace = go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Past Prices',
        line=dict(color=color)
    )

    future_dates = pd.date_range(start=data.index[-1], periods=60, freq='D')
    future_trace = go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines',
        name='Predicted Prices (Next 60 Days)',
        line=dict(color='red', dash='dot')
    )

    layout = go.Layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price (USD)"),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
    )

    fig = go.Figure(data=[past_trace, future_trace], layout=layout)
    return pyo.plot(fig, output_type='div')

# Fetch stock-related news from Google News RSS
def get_stock_news(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock+market"
    feed = feedparser.parse(url)
    
    news_data = []
    for entry in feed.entries[:5]:  # Get top 5 news articles
        news_data.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.get("summary", "Click the link to read more.")
        })

    return news_data

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symbol1 = request.form["symbol1"].upper()
        

        data1, info1, future_prices1, current_price1 = predict_stock(symbol1)
        if data1 is None:
            return render_template("index.html", error="Invalid stock symbol!")

        plot1 = plot_stock(data1, future_prices1, "cyan")
        predicted_price1 = round(future_prices1[-1], 2)
        news1 = get_stock_news(symbol1)

        
      

        return render_template("index.html",
                               symbol1=symbol1, current_price1=current_price1, predicted_price1=predicted_price1, plot1=plot1, info1=info1, news1=news1,
                               )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
