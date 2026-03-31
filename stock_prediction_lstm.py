import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from datetime import datetime
import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

warnings.filterwarnings('ignore')

# Ensure NLTK vader lexicon is downloaded for sentiment analysis demo
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# 1. Fetch Historical Stock Data
def load_data(ticker="AAPL", period="10y", retries=3):
    print(f"Fetching data for {ticker} for the last {period}...")
    
    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if not df.empty:
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise ValueError(f"Could not fetch data for {ticker} after {retries} attempts.")
            time.sleep(2)
            
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. Please check the symbol.")
        
    df.index = pd.to_datetime(df.index)
    # Ensure index is timezone-naive so it plays nice with matplotlib
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df

# 2. Add Sentiment Data
def add_sentiment_data(df):
    print("Generating simulated news sentiment data...")
    np.random.seed(42)
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    
    for i in range(len(df)):
        simulated_score = np.random.normal(0.05, 0.3)
        simulated_score = max(-1, min(1, simulated_score))
        sentiments.append(simulated_score)
        
    df['Sentiment'] = sentiments
    return df

# 3. Preprocess Data for LSTM
def preprocess_data(df, target_col='Close', feature_cols=['Close', 'Sentiment'], sequence_length=60):
    print("Preprocessing data...")
    
    df = df.dropna()
    data = df[feature_cols].values
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    target_idx = feature_cols.index(target_col)

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, target_idx])

    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    dates = df.index[sequence_length:]
    dates_train, dates_test = dates[:split], dates[split:]
    
    print(f"Training shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shape:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, target_idx, dates_train, dates_test

# 4. Build PyTorch LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        dropout_rate = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_model(X_train, y_train, input_size, epochs=10, batch_size=32, device='cpu'):
    print(f"Training on device: {device}")
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # Validation split of 10%
    val_split = int(0.1 * len(X_tensor))
    train_size = len(X_tensor) - val_split
    
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = StockLSTM(input_size=input_size, hidden_size=64, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss_denom = len(train_loader.dataset)
        train_loss = train_loss / train_loss_denom if train_loss_denom > 0 else 0
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss_denom = len(val_loader.dataset)
        val_loss = val_loss / val_loss_denom if val_loss_denom > 0 else 0
        
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model

# 5. Model Evaluation and Plotting
def evaluate_and_plot(model, X_test, y_test, scaler, target_idx, dates_test, num_features, device='cpu'):
    print("Predicting and plotting results...")
    
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    dummy_array = np.zeros((len(predictions), num_features))
    dummy_array[:, target_idx] = predictions.flatten()
    inv_predictions = scaler.inverse_transform(dummy_array)[:, target_idx]
    
    dummy_array_y = np.zeros((len(y_test), num_features))
    dummy_array_y[:, target_idx] = y_test
    inv_y_test = scaler.inverse_transform(dummy_array_y)[:, target_idx]

    plt.figure(figsize=(16,8))
    plt.plot(dates_test, inv_y_test, color='blue', label='Actual Stock Price')
    plt.plot(dates_test, inv_predictions, color='red', label='Predicted Stock Price')
    
    plt.title('Stock Price Prediction with LSTM & Sentiment Analysis')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved successfully to {plot_path}")
    plt.show()

if __name__ == "__main__":
    TICKER = "AAPL"
    PERIOD = "5y" 
    SEQUENCE_LENGTH = 60 
    
    df = load_data(ticker=TICKER, period=PERIOD)
    df = add_sentiment_data(df)
    
    features = ['Close', 'Sentiment']
    
    X_train, X_test, y_train, y_test, scaler, target_idx, dates_train, dates_test = preprocess_data(
        df=df, 
        target_col='Close', 
        feature_cols=features, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    print("Starting Model Training. This may take a minute or two...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = train_model(
        X_train=X_train, 
        y_train=y_train, 
        input_size=len(features),
        epochs=50, 
        batch_size=32,
        device=device
    )
    
    # Concatenate train and test sets to evaluate and plot the full requested period
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    dates_full = list(dates_train) + list(dates_test)
    
    evaluate_and_plot(model, X_full, y_full, scaler, target_idx, dates_full, len(features), device=device)
