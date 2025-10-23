import pandas as pd
import numpy as np

df = pd.read_csv('data/hourly.csv')

df = df.copy()

# Log return
df['logret'] = np.log(df['Close'] / df['Close'].shift(1))

# Lags
for k in [1,2,3,5,8,13]:
    df[f'logret_lag_{k}'] = df['logret'].shift(k)
    df[f'close_lag_{k}'] = df['Close'].shift(k)

# Rolling stats
for w in [5,10,20]:
    df[f'ret_mean_{w}'] = df['logret'].rolling(w).mean()
    df[f'ret_std_{w}'] = df['logret'].rolling(w).std()
    df[f'vol_mean_{w}'] = df['Volume'].rolling(w).mean()

# Moving averages & EMA
df['sma_5'] = df['Close'].rolling(5).mean()
df['sma_20'] = df['Close'].rolling(20).mean()
df['sma_50'] = df['Close'].rolling(50).mean()
df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()

# MACD (12,26,9)
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# RSI (14)
delta = df['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
avg_up = up.ewm(alpha=1/14, adjust=False).mean()
avg_down = down.ewm(alpha=1/14, adjust=False).mean()
rs = avg_up / (avg_down + 1e-12)
df['rsi_14'] = 100 - (100 / (1 + rs))

# Bollinger Bands (20,2)
df['bb_mid'] = df['Close'].rolling(20).mean()
df['bb_std'] = df['Close'].rolling(20).std()
df['bb_high'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_low']  = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

# ATR(14)
high_low = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift(1)).abs()
low_close = (df['Low'] - df['Close'].shift(1)).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
df['atr_pct'] = (df['atr_14'] / df['Close'])

# OBV
df['obv'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()

# normalized interaction
df['close_div_sma20'] = df['Close'] / df['sma_20']
df['zscore_ret_20'] = (df['logret'] - df['ret_mean_20']) / (df['ret_std_20'] + 1e-12)

df['target_reg'] = df['logret'].shift(-1)  # Regression: predict next-day log return
df['target_cls'] = (df['logret'].shift(-1) > 0).astype(int)  # Classification: 1 if up, 0 if down

# Drop NA from initial windows
df = df.dropna().copy()

features_to_keep = ['macd', 'macd_signal', 'rsi_14', 'bb_width', 'close_div_sma20', 'zscore_ret_20', 'atr_pct']
features_to_keep.extend([col for col in df.columns if col.startswith("logret_")])
features_to_keep.extend([col for col in df.columns if col.startswith("ret_mean_")])
features_to_keep.extend([col for col in df.columns if col.startswith("ret_std_")])

total = features_to_keep + ['target_cls', 'target_reg', 'Close']

feature_df = df[total]
feature_df.to_csv("features.csv")