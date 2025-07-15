import signal
import sys
from types import FrameType
from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.logging import logger

app = Flask(__name__)

# Utility function to fetch stock data
def fetch_stock_data(ticker, period="5y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df["Close"]
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return None

# Utility function to resample data to monthly
def resample_monthly(data):
    return data.resample("M").last()

# Utility function to fit SARIMAX model
def fit_sarimax(data):
    try:
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        return model.fit(disp=False)
    except Exception as e:
        logger.error(f"Error fitting SARIMAX model: {str(e)}")
        return None

# Utility function to generate forecast
def forecast(model, steps):
    try:
        return model.forecast(steps=steps)
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return None

@app.route("/current-value/<ticker>")
def get_current_value(ticker):
    logger.info(f"Fetching current value for ticker: {ticker}", extra={"logField": "current-value"})
    try:
        #stock = yf.Ticker(f"{ticker}.KA")
        #current_price = stock.history(period="1d")["Close"].iloc[-1]
        df = fetch_stock_data(f"{ticker}.KA")
        current_price = df.iloc[-1]
        return jsonify({"currentPrice": current_price})
    except Exception as e:
        logger.error(f"Error fetching current value for {ticker}: {str(e)}")
        return jsonify({"error": f"Invalid ticker symbol: {ticker}"}), 400

@app.route("/forecast/<ticker>")
def get_forecast(ticker):
    logger.info(f"Generating forecast for ticker: {ticker}", extra={"logField": "forecast"})
    try:
        df = fetch_stock_data(f"{ticker}.KA")
        if df is None:
            return jsonify({"error": f"Invalid ticker symbol: {ticker}"}), 400

        current_price = df.iloc[-1]
        monthly_data = resample_monthly(df)
        model = fit_sarimax(monthly_data)
        if model is None:
            return jsonify({"error": "Failed to fit forecasting model"}), 500

        forecast_steps = 6
        future_forecast = forecast(model, forecast_steps)
        if future_forecast is None:
            return jsonify({"error": "Failed to generate forecast"}), 500

        future_dates = pd.date_range(monthly_data.index[-1], periods=forecast_steps + 1, freq="M")[1:]
        forecast_data = [
            {"Date": date.strftime("%d-%m-%Y"), "Forecast": value}
            for date, value in zip(future_dates, future_forecast)
        ]

        historical_dates = [date.strftime("%d-%m-%Y") for date in monthly_data.index]
        historical_values = monthly_data.tolist()

        return jsonify({
            "currentPrice": current_price,
            "forecast": forecast_data,
            "historicalDates": historical_dates,
            "historicalValues": historical_values,
            "forecastDates": [d["Date"] for d in forecast_data],
            "forecastValues": [d["Forecast"] for d in forecast_data]
        })
    except Exception as e:
        logger.error(f"Error in forecast for {ticker}: {str(e)}")
        return jsonify({"error": str(e)}), 500

def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")
    from utils.logging import flush
    flush()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    app.run(host="localhost", port=8080, debug=True)
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
