import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

# App Configuration
st.set_page_config(page_title="NSE Pairs Trader", layout="wide")

# --- UI TABS ---
tab_app, tab_instr = st.tabs(["🚀 Trading Dashboard", "📖 How to Use"])

with tab_instr:
    st.header("Instructions for Pair Trading")
    st.markdown("""
    ### **1. Select Your Pair**
    * Use stock symbols with the `.NS` suffix (e.g., `TCS.NS`, `INFY.NS`).
    * **Tip:** Choose stocks in the same sector (Banks, IT, Auto) for better cointegration.
    
    ### **2. Understand the Signals**
    * **BUY SPREAD**: The spread is too low. **Buy Stock 2** and **Sell Stock 1**.
    * **SELL SPREAD**: The spread is too high. **Sell Stock 2** and **Buy Stock 1**.
    * **EXIT**: The prices have converged. Close both positions to book profit.
    
    ### **3. Review Costs**
    * The 'Est. Charges' includes **STT (0.025% on sell)**, **Brokerage (capped at ₹20)**, and **18% GST** as per Indian norms.
    
    ### **4. Risk Management**
    * Only trade if the **Cointegration P-Value < 0.05**. If it is higher, the stocks do not move together reliably.
    """)

with tab_app:
    st.sidebar.header("Strategy Settings")
    t1 = st.sidebar.text_input("Stock 1 (e.g. HDFCBANK.NS)", "HDFCBANK.NS")
    t2 = st.sidebar.text_input("Stock 2 (e.g. ICICIBANK.NS)", "ICICIBANK.NS")
    days = st.sidebar.slider("Historical Lookback (Days)", 100, 730, 365)
    z_thresh = st.sidebar.slider("Z-Score Entry Threshold", 1.5, 3.0, 2.0)

    # Data Fetching Logic
    @st.cache_data
    def load_nse_data(ticker1, ticker2, lookback):
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=lookback)
        raw = yf.download([ticker1, ticker2], start=start, end=end)
        
        # Safe MultiIndex indexing for yfinance
        if 'Close' in raw.columns:
            df = raw['Close'].dropna()
        else:
            df = raw['Adj Close'].dropna()
        return df

    try:
        df = load_nse_data(t1, t2, days)
        
        # Stats Calculation
        S1, S2 = df[t1], df[t2]
        score, pvalue, _ = coint(S1, S2)
        
        # Hedge Ratio (Beta) calculation
        model = sm.OLS(S2, sm.add_constant(S1)).fit()
        beta = model.params[t1]
        spread = S2 - (beta * S1)
        z_score = (spread - spread.mean()) / spread.std()

        # Display Top Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Cointegration (P-Value)", f"{pvalue:.4f}", delta="Good" if pvalue < 0.05 else "Weak", delta_color="normal")
        m2.metric("Current Z-Score", f"{z_score.iloc[-1]:.2f}")
        
        # Indian Cost Estimator (Simplified Zerodha model)
        price_val = S2.iloc[-1]
        charges = (min(20, 0.0003 * price_val) + (0.00025 * price_val) + (0.0000345 * price_val)) * 1.18
        m3.metric("Est. Charges/Unit", f"₹{charges:.2f}")

        # Current Signal UI
        st.divider()
        current_z = z_score.iloc[-1]
        if current_z < -z_thresh:
            st.success(f"### 🟢 SIGNAL: BUY SPREAD\n**Strategy:** Buy {t2} | Sell {t1}")
        elif current_z > z_thresh:
            st.error(f"### 🔴 SIGNAL: SELL SPREAD\n**Strategy:** Sell {t2} | Buy {t1}")
        elif abs(current_z) < 0.5:
            st.warning(f"### 🟡 SIGNAL: EXIT\n**Strategy:** Square off all positions.")
        else:
            st.info("### ⚪ SIGNAL: WAIT\nSpread is in neutral territory.")

        # Visual Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(z_score, color='silver', label='Z-Score')
        ax.axhline(z_thresh, color='red', linestyle='--')
        ax.axhline(-z_thresh, color='green', linestyle='--')
        ax.axhline(0, color='black', alpha=0.5)
        # Highlight entry points
        ax.scatter(z_score[z_score > z_thresh].index, z_score[z_score > z_thresh], color='red', marker='v')
        ax.scatter(z_score[z_score < -z_thresh].index, z_score[z_score < -z_thresh], color='green', marker='^')
        st.pyplot(fig)

        # Historical Dataframe with Style
        st.subheader("Recent Signal History")
        history = pd.DataFrame({'Z-Score': z_score, 'Stock1': S1, 'Stock2': S2})
        history['Action'] = "Wait"
        history.loc[history['Z-Score'] < -z_thresh, 'Action'] = "BUY SPREAD"
        history.loc[history['Z-Score'] > z_thresh, 'Action'] = "SELL SPREAD"
        history.loc[history['Z-Score'].abs() < 0.5, 'Action'] = "EXIT"

        def style_rows(row):
            if row['Action'] == "BUY SPREAD": return ['background-color: #d4edda'] * 4
            if row['Action'] == "SELL SPREAD": return ['background-color: #f8d7da'] * 4
            if row['Action'] == "EXIT": return ['background-color: #fff3cd'] * 4
            return [''] * 4

        st.dataframe(history.tail(15).style.apply(style_rows, axis=1), use_container_width=True)

    except Exception as e:
        st.error(f"Please ensure tickers are correct (e.g., RELIANCE.NS). Error: {e}")
