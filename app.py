# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# import torch
# import numpy as np
# import traceback

# # Import functions from existing script
# from stock_prediction_lstm import load_data, add_sentiment_data, preprocess_data, train_model

# app = Flask(__name__)
# CORS(app)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         ticker = data.get('ticker', 'AAPL')
#         period = data.get('period', '5y')
#         epochs = 50  # Increased for better convergence
#         seq_length = 100 # Increased for more context
        
#         print(f"Prediction requested for Ticker={ticker}, Period={period}")
        
#         # 1. Fetch data
#         df = load_data(ticker=ticker, period=period)
        
#         # Determine appropriate sequence length based on data size
#         data_len = len(df)
#         if data_len < seq_length + 10:
#             seq_length = max(5, int(data_len * 0.2))
#             print(f"Adjusted sequence length to {seq_length} due to short data period (length: {data_len})")
            
#         # 2. Add sentiment
#         df = add_sentiment_data(df)
        
#         # 3. Preprocess
#         features = ['Close', 'Sentiment']
#         X_train, X_test, y_train, y_test, scaler, target_idx, dates_train, dates_test = preprocess_data(
#             df=df, target_col='Close', feature_cols=features, sequence_length=seq_length
#         )
        
#         # If not enough data properly splits into X_train, simply fallback
#         if len(X_train) == 0:
#             raise ValueError(f"Period {period} provides too little data ({data_len} days) to train the model properly.")
            
#         # 4. Train Model
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = train_model(
#             X_train=X_train, y_train=y_train, input_size=len(features),
#             epochs=epochs, batch_size=32, device=device
#         )
        
#         # 5. Evaluate
#         model.eval()
        
#         # Concatenate train and test sets to show the full period requested by the user
#         X_full = np.concatenate((X_train, X_test), axis=0)
#         y_full = np.concatenate((y_train, y_test), axis=0)
#         dates_full = list(dates_train) + list(dates_test)
        
#         X_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)
#         with torch.no_grad():
#             predictions = model(X_tensor).cpu().numpy()
            
#         dummy_array = np.zeros((len(predictions), len(features)))
#         dummy_array[:, target_idx] = predictions.flatten()
#         inv_predictions = scaler.inverse_transform(dummy_array)[:, target_idx]
        
#         dummy_array_y = np.zeros((len(y_full), len(features)))
#         dummy_array_y[:, target_idx] = y_full
#         inv_y_test = scaler.inverse_transform(dummy_array_y)[:, target_idx]
        
#         # Ensure we return valid JS-compatible date strings
#         dates_str = [d.strftime('%Y-%m-%d') for d in dates_full]
        
#         return jsonify({
#             'status': 'success',
#             'dates': dates_str,
#             'actual': inv_y_test.tolist(),
#             'predicted': inv_predictions.tolist()
#         })
        
#     except Exception as e:
#         print("Error during prediction:")
#         traceback.print_exc()
#         return jsonify({'status': 'error', 'message': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import traceback
import os  # <--- ADDED THIS

# Import functions from existing script
from stock_prediction_lstm import load_data, add_sentiment_data, preprocess_data, train_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL')
        period = data.get('period', '5y')
        epochs = 50 
        seq_length = 100 
        
        print(f"Prediction requested for Ticker={ticker}, Period={period}")
        
        # 1. Fetch data
        df = load_data(ticker=ticker, period=period)
        
        # Determine appropriate sequence length based on data size
        data_len = len(df)
        if data_len < seq_length + 10:
            seq_length = max(5, int(data_len * 0.2))
            print(f"Adjusted sequence length to {seq_length} due to short data period (length: {data_len})")
            
        # 2. Add sentiment
        df = add_sentiment_data(df)
        
        # 3. Preprocess
        features = ['Close', 'Sentiment']
        X_train, X_test, y_train, y_test, scaler, target_idx, dates_train, dates_test = preprocess_data(
            df=df, target_col='Close', feature_cols=features, sequence_length=seq_length
        )
        
        if len(X_train) == 0:
            raise ValueError(f"Period {period} provides too little data ({data_len} days) to train the model properly.")
            
        # 4. Train Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = train_model(
            X_train=X_train, y_train=y_train, input_size=len(features),
            epochs=epochs, batch_size=32, device=device
        )
        
        # 5. Evaluate
        model.eval()
        
        X_full = np.concatenate((X_train, X_test), axis=0)
        y_full = np.concatenate((y_train, y_test), axis=0)
        dates_full = list(dates_train) + list(dates_test)
        
        X_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
            
        dummy_array = np.zeros((len(predictions), len(features)))
        dummy_array[:, target_idx] = predictions.flatten()
        inv_predictions = scaler.inverse_transform(dummy_array)[:, target_idx]
        
        dummy_array_y = np.zeros((len(y_full), len(features)))
        dummy_array_y[:, target_idx] = y_full
        inv_y_test = scaler.inverse_transform(dummy_array_y)[:, target_idx]
        
        dates_str = [d.strftime('%Y-%m-%d') for d in dates_full]
        
        return jsonify({
            'status': 'success',
            'dates': dates_str,
            'actual': inv_y_test.tolist(),
            'predicted': inv_predictions.tolist()
        })
        
    except Exception as e:
        print("Error during prediction:")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- UPDATED THIS SECTION FOR RENDER ---
if __name__ == '__main__':
    # Use the port Render provides, or default to 10000 locally
    port = int(os.environ.get("PORT", 10000))
    # Must use 0.0.0.0 for the host so Render can route traffic to it
    app.run(host='0.0.0.0', port=port)