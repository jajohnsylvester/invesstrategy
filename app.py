import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

# 1. Fetch Indian Stock Data (Suffix .NS for National Stock Exchange)
tickers = ['HDFCBANK.NS', 'ICICIBANK.NS']
data = yf.download(tickers, start='2023-01-01', end='2025-01-01')['Adj Close']

# 2. Check for Cointegration
# Null Hypothesis: No cointegration. If p-value < 0.05, they are cointegrated.
score, pvalue, _ = coint(data[tickers[0]], data[tickers[1]])
print(f"Cointegration P-Value: {pvalue:.4f}")

# 3. Calculate the Spread using Linear Regression (Hedge Ratio)
# Spread = Stock1 - (Beta * Stock2)
S1 = data[tickers[0]]
S2 = data[tickers[1]]
S1_with_const = sm.add_constant(S1)
model = sm.OLS(S2, S1_with_const).fit()
beta = model.params[tickers[0]]
spread = S2 - beta * S1

# 4. Generate Trading Signals (Z-Score)
def calculate_zscore(series):
    return (series - series.mean()) / np.std(series)

z_score = calculate_zscore(spread)

# 5. Plotting the Strategy
plt.figure(figsize=(12, 6))
z_score.plot()
plt.axhline(z_score.mean(), color='black')
plt.axhline(2.0, color='red', linestyle='--')   # Sell threshold
plt.axhline(-2.0, color='green', linestyle='--') # Buy threshold
plt.title(f"Z-Score Spread: {tickers[0]} vs {tickers[1]}")
plt.show()
