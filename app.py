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
    ### **1. Setup**
    * Enter NSE symbols with `.NS` (e.g., `RELIANCE.NS`, `ONGC.NS`).
    * Use the **Z-Score Threshold** to control risk (Standard is 2.0).
    
    ### **2. Execution Guide**
    * <span style='color:green; font-weight:bold;'>BUY SPREAD</span>: Buy Stock 2 and Short Stock 1.
    * <span style='color:red; font-weight:bold;'>SELL SPREAD</span>: Short Stock 2 and Buy Stock 1.
    * <span style='color:orange; font-weight:bold;'>EXIT</span>: Close both positions immediately to realize profit.
    
    ### **3. Charges**
    * Estimates include **STT (0.025%)**, **Brokerage (₹20 cap)**, and **18% GST**.
    """, unsafe_allow_html=True)

with tab_app:
    st.sidebar.header("Market Settings")
    t1 = st.sidebar.text_input("Stock 1 (Hedge)", "HDFCBANK.NS")
    t2 = st.sidebar.text_input("Stock 2 (Target)", "ICICIBANK.NS")
    days = st.sidebar.slider("Lookback Days", 100, 730, 365)
    z_thresh = st.sidebar.slider("Z-Score Threshold", 1.5, 3.0, 2.0)

    @st.cache_data
    def load_data(ticker1, ticker2, lookback):
        raw = yf.download([ticker1, ticker2], period=f"{lookback}d")
        return raw['Close'].dropna() if 'Close' in raw.columns else raw['Adj Close'].dropna()

    try:
        df = load_data(t1, t2, days)
        S1, S2 = df[t1], df[t2]
        
        # Statistics
        _, pvalue, _ = coint(S1, S2)
        model = sm.OLS(S2, sm.add_constant(S1)).fit()
        beta = model.params[t1]
        spread = S2 - (beta * S1)
        z_score = (spread - spread.mean()) / spread.std()
        
        # Signal Logic
        curr_z = z_score.iloc[-1]
        
        st.subheader("🎯 Live Execution Signal")
        
        if curr_z < -z_thresh:
            st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:#d4edda; border:2px solid #28a745">
                    <h2 style="color:#155724; margin:0;">ACTION: BUY SPREAD</h2>
                    <p style="font-size:20px; color:#155724;">
                        🟢 <b>BUY {t2}</b> (Quantity: 100)<br>
                        🔴 <b>SELL {t1}</b> (Quantity: {round(100*beta)})
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        elif curr_z > z_thresh:
            st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:#f8d7da; border:2px solid #dc3545">
                    <h2 style="color:#721c24; margin:0;">ACTION: SELL SPREAD</h2>
                    <p style="font-size:20px; color:#721c24;">
                        🔴 <b>SELL {t2}</b> (Quantity: 100)<br>
                        🟢 <b>BUY {t1}</b> (Quantity: {round(100*beta)})
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        elif abs(curr_z) < 0.5:
            st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:#fff3cd; border:2px solid #ffc107">
                    <h2 style="color:#856404; margin:0;">ACTION: EXIT SIGNAL</h2>
                    <p style="font-size:20px; color:#856404;">
                        ⚠️ <b>SQUARE OFF ALL POSITIONS</b><br>
                        Sell your holdings in {t2} and Buy back {t1} (or vice-versa).
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⌛ **SIGNAL: NEUTRAL** - Wait for Z-Score to hit thresholds.")

        # Metrics & Visuals
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("P-Value (Cointegration)", f"{pvalue:.4f}", help="Below 0.05 is ideal")
            st.metric("Hedge Ratio (Beta)", f"{beta:.3f}")
            price_t2 = S2.iloc[-1]
            charges = (min(20, 0.0003 * price_t2) + (0.00025 * price_t2) + (0.0000345 * price_t2)) * 1.18
            st.metric("Est. Taxes/Unit", f"₹{charges:.2f}")

        with col2:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(z_score, color='gray', alpha=0.4)
            ax.axhline(z_thresh, color='red', linestyle='--')
            ax.axhline(-z_thresh, color='green', linestyle='--')
            ax.axhline(0, color='black', linewidth=1)
            ax.fill_between(z_score.index, z_thresh, z_score, where=(z_score >= z_thresh), color='red', alpha=0.3)
            ax.fill_between(z_score.index, -z_thresh, z_score, where=(z_score <= -z_thresh), color='green', alpha=0.3)
            st.pyplot(fig)

        # Historical Log with Color Coding
        st.subheader("📜 Recent Trade Logs")
        history = pd.DataFrame({'Z-Score': z_score, t1: S1, t2: S2})
        history['Action'] = "Neutral"
        history.loc[history['Z-Score'] < -z_thresh, 'Action'] = "BUY SPREAD"
        history.loc[history['Z-Score'] > z_thresh, 'Action'] = "SELL SPREAD"
        history.loc[history['Z-Score'].abs() < 0.5, 'Action'] = "EXIT"

        def color_map(val):
            if val == "BUY SPREAD": return 'background-color: #d4edda; color: #155724'
            if val == "SELL SPREAD": return 'background-color: #f8d7da; color: #721c24'
            if val == "EXIT": return 'background-color: #fff3cd; color: #856404'
            return ''

        st.dataframe(history.tail(20).style.applymap(color_map, subset=['Action']), use_container_width=True)

    except Exception as e:
        st.warning(f"Waiting for valid NSE tickers... (Ensure you use .NS suffix)")
