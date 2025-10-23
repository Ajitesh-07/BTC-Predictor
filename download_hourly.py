import yfinance as yf

# Ticker for Bitcoin-USD
ticker = 'BTC-USD'

print("Fetching hourly BTC data for the last 730 days...")
data_hourly = yf.download(
    tickers=ticker,
    period='730d',  # Max allowed period
    interval='1h'   # '1h' for hourly
)
data_hourly.to_csv('data/hourly.csv')