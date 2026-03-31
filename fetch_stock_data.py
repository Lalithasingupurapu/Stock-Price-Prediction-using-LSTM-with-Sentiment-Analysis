import yfinance as yf
import pandas as pd

def save_data_to_csv(ticker, filename, period="5y"):
    print(f"Fetching {period} of data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty:
            df.to_csv(filename)
            print(f"Success! Data saved to {filename}")
        else:
            print(f"Warning: No data found for {ticker}")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

if __name__ == "__main__":
    print("Starting stock data download...")
    save_data_to_csv("TCS.NS", "TCS_stock_data.csv")
    save_data_to_csv("MSFT", "MSFT_stock_data.csv")
    print("All downloads complete!")
