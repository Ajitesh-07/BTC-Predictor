import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import joblib

from model import LSTMRegressor
from train_lstm import TimeSeriesDataset

TYPE = 'reg' 

SEQ_LEN = 1
TEST_SPLIT = 0.2

HIDDEN_SIZE = 16
DENSE_SIZE = 32
NUM_LAYERS = 2

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")

    try:
        df_features = pd.read_csv('features.csv', index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        return
    
    if 'Close' not in df_features.columns:
        print("Error: 'Close' column not found in 'features.csv'. Cannot reconstruct price.")
        return

    # MODIFIED: Explicitly exclude 'Close' from the list of features for the model.
    features_for_model = [col for col in df_features.columns if col not in ['target_cls', 'target_reg', 'Close']]
    num_features = len(features_for_model)
    print(f"Model configured for {num_features} features.")

    # Initialize model architecture
    model = LSTMRegressor(num_features, HIDDEN_SIZE, DENSE_SIZE, NUM_LAYERS)
    
    try:
        # Load the saved weights
        model.load_state_dict(torch.load('best_model_reg.pth', map_location=device))
    except FileNotFoundError:
        print("Error: 'best_model_reg.pth' not found. Please train the regression model first.")
        return
        
    model.to(device)
    model.eval() # Set model to evaluation mode (very important!)

    # --- 2. Load and Prepare Test Data ---
    target_col = 'target_reg'
    
    X = df_features[features_for_model]
    y = df_features[target_col]

    split_index = int(len(df_features) * (1 - TEST_SPLIT))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    scaler = joblib.load('scaler.pkl')
    X_test_scaled = scaler.transform(X_test)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor, seq_len=SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Generating predictions on the test set...")
    all_preds_logret = []
    all_actuals_logret = []

    with torch.no_grad():
        for features_seq, targets in test_loader:
            features_seq = features_seq.to(device)
            outputs = model(features_seq).squeeze()
            
            all_preds_logret.extend(outputs.cpu().numpy())
            all_actuals_logret.extend(targets.cpu().numpy())

    last_train_price = df_features['Close'].iloc[split_index - 1]
    
    test_dates = df_features.index[split_index + SEQ_LEN - 1 : split_index + SEQ_LEN - 1 + len(all_preds_logret)]

    actual_prices = df_features['Close'].iloc[split_index + SEQ_LEN - 1 : split_index + SEQ_LEN - 1 + len(all_actuals_logret)]

    predicted_prices = []
    current_price = last_train_price
    for log_return in all_preds_logret:
        next_price = current_price * np.exp(log_return)
        predicted_prices.append(next_price)
        current_price = next_price

    print("Plotting results...")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=actual_prices,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=predicted_prices,
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Actual vs. Predicted Stock Price (Test Set)',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        hovermode='x unified'
    )

    fig.show()

if __name__ == "__main__":
    main()

