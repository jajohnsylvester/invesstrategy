import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

st.set_page_config(page_title="NSE Pair Trading Tool", layout="wide")

st.title("📈 Indian Stock Market: Pair Trading Tool")
st.sidebar.header("Strategy Settings")

# 1. User Inputs
ticker1 = st.sidebar.text_input("First Stock (NSE)", "HDFCBANK.NS")
ticker2 = st.sidebar.text_input("Second Stock (NSE)", "ICICIBANK.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# 2. Data Fetching with KeyError Fix
@st.cache_data
def get_data(t1, t2, start, end):
    tickers = [t1, t2]
    raw = yf.download(tickers, start=start, end=end)
    
    # Handle yfinance MultiIndex specifically to avoid KeyError
    if 'Close' in raw.columns:
        df = raw['Close'].dropna()
    elif 'Adj Close' in raw.columns:
        df = raw['Adj Close'].dropna()
    else:
        st.error("Could not find 'Close' or 'Adj Close' columns.")
        return None
    return df

data = get_data(ticker1, ticker2, start_date, end_date)

if data is not None and not data.empty:
    # 3. Statistical Analysis
    S1, S2 = data[ticker1], data[ticker2]
    
    # Check for Cointegration
    _, p_value, _ = coint(S1, S2)
    st.metric("Cointegration P-Value", f"{p_value:.4f}")
    if p_value > 0.05:
        st.warning("⚠️ These stocks are NOT significantly cointegrated. Signals may be unreliable.")

    # Calculate Spread & Z-Score
    model = sm.OLS(S2, sm.add_constant(S1)).fit()
    beta = model.params[ticker1]
    spread = S2 - (beta * S1)
    z_score = (spread - spread.mean()) / spread.std()

    # 4. Indian Charges Calculation (Intraday)
    def get_charges(price, qty=100, side='sell'):
        turnover = price * qty
        brokerage = min(20, 0.0003 * turnover)  # ₹20 or 0.03%
        stt = 0.00025 * turnover if side == 'sell' else 0 # 0.025% on sell side only
        txn_fee = 0.0000345 * turnover
        gst = 0.18 * (brokerage + txn_fee)
        return brokerage + stt + txn_fee + gst

    # 5. Signal Generation
    results = pd.DataFrame({'Z': z_score, ticker1: S1, ticker2: S2})
    results['Signal'] = "Neutral"
    results.loc[results['Z'] < -2.0, 'Signal'] = f"BUY SPREAD (Long {ticker2})"
    results.loc[results['Z'] > 2.0, 'Signal'] = f"SELL SPREAD (Short {ticker2})"
    results.loc[results['Z'].abs() < 0.5, 'Signal'] = "EXIT"

    # 6. Display Output
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Z-Score Thresholds")
        fig, ax = plt.subplots()
        ax.plot(z_score, label='Z-Score', color='royalblue')
        ax.axhline(2, color='red', linestyle='--', label='Sell Threshold')
        ax.axhline(-2, color='green', linestyle='--', label='Buy Threshold')
        ax.axhline(0, color='black', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("Current Market Signal")
        latest_signal = results['Signal'].iloc[-1]
        latest_price = S2.iloc[-1]
        cost = get_charges(latest_price, side='sell' if "SELL" in latest_signal else 'buy')
        
        st.info(f"Latest Signal: **{latest_signal}**")
        st.write(f"Estimated NSE Charges for 100 shares of {ticker2}: **₹{cost:.2f}**")
        st.dataframe(results.tail(10))

else:
    st.error("No data found. Ensure the tickers have the '.NS' suffix.")
