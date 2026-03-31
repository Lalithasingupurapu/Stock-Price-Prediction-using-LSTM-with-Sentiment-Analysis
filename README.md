# Stock Price Prediction using LSTM and Sentiment Analysis

This project demonstrates an implementation of a Long Short-Term Memory (LSTM) Neural Network predicting the stock price of Apple (`AAPL`) using both numerical (Historical `Close` Prices) and NLP-derived (Simulated Sentiment Scores) inputs.

It relies on TensorFlow, Keras, `yfinance` for fetching the historical prices, and uses NLTK VADER as a demonstration structure for parsing sentiment.

## Important Note regarding Sentiment
Real sentiment data fetched dynamically across several years generally requires a paid API for news scraping (e.g. Bloomberg API, Finnhub, or NewsAPI on a paid tier). To demonstrate a complete, running pipeline out-of-the-box, this script generates statistically-skewed sentiment placeholders for historical days, but integrates the **exact logic** you would use once you hook up a real news text scraper or API. 

## Getting Started

### 1. Prerequisites
Ensure you have Python 3 installed. You'll need the libraries present in the requirements list.

### 2. Installations
You can install the necessary dependencies via:
```bash
pip install -r requirements.txt
```

*(Note for NLTK: The script will automatically download the `vader_lexicon` dictionary upon first execution)*

### 3. Execution
To train the model and generate a prediction chart, run:
```bash
python stock_prediction_lstm.py
```

## How It Works

1. **Data Loading:** The script calls `yfinance.Ticker("AAPL").history(period="5y")` to get standard historical Open, High, Low, Close, and Volume details over the last 5 years.
2. **Sentiment Generation:** An `add_sentiment_data` function populates each date with a Sentiment Compound Score. VADER usage is shown.
3. **Preprocessing:** 
   - Uses `MinMaxScaler` to scale Price and Sentiment fields between $0$ and $1$ to improve LSTM convergence.
   - Restructures sequential 2D tabular data into a 3D matrix expected by `LSTM`, applying a Sequence Length (Lookback Window) of `60` days (i.e. we use the past 60 days to predict "tomorrows" price).
4. **Model Architecture:**
   - 2 LSTM layers structure with 50 units.
   - 2 `Dropout` layers with a 0.2 rate strictly to prevent overfitting.
   - 1 Dense layer to output the predicted next target value.
5. **Evaluation:** The final predictions inverse-transform the scaler to bring the values identically scaling back to real prices (USD) and outputs a `prediction_plot.png` contrasting True Price vs Estimated Price.
