import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np 

try:
    df = pd.read_csv('./features.csv', index_col=0) 
except FileNotFoundError:
    print("Error: 'features.csv' not found.")
    print("Please make sure the file is in the same directory as this script.")
    exit()

target = 'target_reg'
features = [col for col in df.columns if col not in ['target_cls', 'target_reg', 'Close']]

X = df[features]
y = df[target]

if X.empty:
    print("Error: No features found. Check your 'features.csv' file.")
    exit()

split_percentage = 0.9
split_index = int(len(df) * split_percentage)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Total samples: {len(df)}")
print(f"Training samples: {len(X_train)} (80%)")
print(f"Testing samples: {len(X_test)} (20%)")
print("-" * 30)

params = {
    'objective': 'reg:squarederror', 
    'eval_metric': 'rmse',
    'use_label_encoder': False,
    'random_state': 42,
    'device': 'cuda', 
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 7
}

model = xgb.XGBRegressor(**params)

print("Training XGBoost regressor...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("Training complete.")
print("-" * 30)

print("Evaluating model on test data...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test Set RMSE: {rmse:.4f}")
print(f"(Lower is better. This is the avg error in the same units as your target)")

r2 = r2_score(y_test, y_pred)
print(f"Test Set R-squared (R2): {r2:.4f}")
print(f"(Closer to 1.0 is better. This shows how much variance is explained)")
print("-" * 30)

print("Plotting Actual vs. Predicted Prices...")

actual_prices = df.loc[y_test.index, 'Close']

last_train_price = df.loc[X_train.index[-1], 'Close']

pred_log_returns = pd.Series(y_pred, index=y_test.index)

pred_simple_returns = np.exp(pred_log_returns)

predicted_prices = last_train_price * pred_simple_returns.cumprod()

plt.figure(figsize=(15, 7))
plt.plot(actual_prices.index, actual_prices, label='Actual Price', color='blue', alpha=0.7)
plt.plot(predicted_prices.index, predicted_prices, label='Predicted Price', color='red', linestyle='--')

plt.title('BTC Price: Actual vs. Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Plotting Actual vs. Predicted Log Returns...")

plt.figure(figsize=(15, 7))
# y_test and y_pred are both log returns, so we can plot them directly
plt.plot(y_test.index, y_test, label='Actual Log Return', color='blue', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted Log Return', color='red', linestyle='--')

plt.title('Log Returns: Actual vs. Predicted')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()