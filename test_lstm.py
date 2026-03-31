import torch
import torch.nn as nn
from stock_prediction_lstm import load_data, add_sentiment_data, preprocess_data, train_model
import numpy as np

features = ['Close', 'Sentiment']
df = load_data('AAPL', '5y')
df = add_sentiment_data(df)
X_train, X_test, y_train, y_test, scaler, target_idx, dates_train, dates_test = preprocess_data(df, target_col='Close', feature_cols=features, sequence_length=60)

print("Test Set variance:", np.var(y_test))

# Let's see original evaluate to check if it's flat
model = train_model(X_train, y_train, input_size=len(features), epochs=15, batch_size=32, device='cpu')
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
print("Prediction variance:", np.var(preds))
print("Prediction range:", np.min(preds), np.max(preds))
print("Actual range:", np.min(y_test), np.max(y_test))
