import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Fetch Data with improved indexing
tickers = ['HDFCBANK.NS', 'ICICIBANK.NS']
raw_data = yf.download(tickers, start='2023-01-01', end='2025-01-01')

# This fixes the KeyError by ensuring we grab the 'Close' column safely
if 'Close' in raw_data.columns:
    data = raw_data['Close']
elif 'Adj Close' in raw_data.columns:
    data = raw_data['Adj Close']
else:
    raise ValueError("Could not find Close or Adj Close in data")

# Drop any missing values to prevent math errors
data = data.dropna()

# 2. Strategy Logic (Rest of the code remains similar)
S1 = data['HDFCBANK.NS']
S2 = data['ICICIBANK.NS']

# Add constant for OLS
S1_with_const = sm.add_constant(S1)
model = sm.OLS(S2, S1_with_const).fit()
beta = model.params['HDFCBANK.NS']

spread = S2 - (beta * S1)
z_score = (spread - spread.mean()) / np.std(spread)

# 3. Create Results DataFrame
df = pd.DataFrame({
    'ICICI': S2,
    'HDFC': S1,
    'Z': z_score
})

# 4. Generate Signal Column
df['Signal'] = 'Wait'
df.loc[df['Z'] < -2.0, 'Signal'] = 'BUY SPREAD (Long ICICI, Short HDFC)'
df.loc[df['Z'] > 2.0, 'Signal'] = 'SELL SPREAD (Short ICICI, Long HDFC)'
df.loc[df['Z'].abs() < 0.5, 'Signal'] = 'EXIT'

print(df.tail(10))
