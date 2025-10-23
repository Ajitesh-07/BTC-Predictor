import yfinance as yf
import pandas as pd
import os

TICKER = 'BTC-USD'
START = '2015-01-01'
END = None

print('downloading data...')
os.makedirs('data/', exist_ok=True)
df = yf.download(TICKER, START, END, progress=True)
df.to_csv('data/2015_curr.csv')
print("Saved to data/2015_curr.csv")