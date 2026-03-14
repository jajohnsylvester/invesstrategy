import streamlit as st
import yfinance as yf
import pandas as pd
import math

# --- CONFIGURATION & BENCHMARKS (MARCH 2026) ---
INDIA_BOND_YIELD = 6.68 

st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Strategy Dashboard")
st.markdown("---")

# --- DATA FETCHING ENGINE ---
@st.cache_data(ttl=3600)
def get_stock_metrics(tickers):
    data_list = []
    for ticker in tickers:
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            # Key Variables
            pe = info.get('trailingPE', 0)
            eps_growth = info.get('earningsGrowth', 0) * 100
            roic = info.get('returnOnCapitalEmployed', info.get('returnOnEquity', 0) * 100)
            ey = (1/pe * 100) if pe > 0 else 0
            
            data_list.append({
                "Ticker": ticker,
                "Price": info.get('currentPrice', 0),
                "P/E": round(pe, 2),
                "ROIC %": round(roic, 2),
                "EY %": round(ey, 2),
                "Growth %": round(eps_growth, 2),
                "PEG": round(pe / eps_growth, 2) if eps_growth > 0 else "N/A"
            })
        except:
            continue
    return pd.DataFrame(data_list)

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs(["Coffee Can Portfolio", "Magic Formula", "Beating the Market"])

# --- TAB 1: COFFEE CAN PORTFOLIO ---
with tab1:
    st.header("Coffee Can Portfolio")
    st.subheader("Strategy: The 'Buy and Forget' Approach")
    st.markdown("""
    **Guidelines:**
    * **Condition:** Companies with Revenue Growth > 10% and ROCE > 15% consistently for 10 years.
    * **How to Use:** Buy high-quality 'moat' businesses and hold for a minimum of 10 years.
    * **When to Sell:** Only if there is a breakdown in corporate governance or the business model is permanently disrupted.
    """)
    
    # Watchlist for Coffee Can
    cc_data = get_stock_metrics(["TITAN.NS", "PIDILITIND.NS", "ASIANPAINT.NS", "NESTLEIND.NS"])
    st.dataframe(cc_data, use_container_width=True)

# --- TAB 2: MAGIC FORMULA ---
with tab2:
    st.header("Magic Formula")
    st.subheader("Strategy: Joel Greenblatt’s High ROIC + High Yield")
    st.markdown("""
    **Guidelines:**
    * **Condition:** Rank stocks by **Earnings Yield** (Cheapness) and **Return on Capital** (Quality).
    * **How to Use:** Buy the top 20-30 ranked stocks. This strategy beats the market by buying good businesses at bargain prices.
    * **When to Sell:** Rebalance the entire portfolio once every year. Sell losers just before 12 months (tax benefit) and winners just after.
    """)
    
    mf_data = get_stock_metrics(["ITC.NS", "COALINDIA.NS", "HCLTECH.NS", "BAJAJ-AUTO.NS"])
    st.dataframe(mf_data, use_container_width=True)

# --- TAB 3: BEATING THE MARKET ---
with tab3:
    st.header("Beating the Market")
    st.subheader("Strategy: Peter Lynch’s Growth at a Reasonable Price (GARP)")
    st.markdown(f"""
    **Guidelines:**
    * **Condition:** PEG Ratio < 1.0 and Earnings Yield > Bond Yield ({INDIA_BOND_YIELD}%).
    * **How to Use:** Identify 'Fast Growers' (20-25% growth) that the market has undervalued. Use your personal 'investor's edge'.
    * **When to Sell:** * **Stalwarts:** Sell after 30-50% gain.
        * **Fast Growers:** Sell if P/E exceeds 2x the Growth Rate or the story changes.
    """)
    
    lynch_data = get_stock_metrics(["CDSL.NS", "RELIANCE.NS", "TATASTEEL.NS", "HAL.NS"])
    st.dataframe(lynch_data, use_container_width=True)

# --- SIDEBAR ---
st.sidebar.header("Dashboard Metrics")
st.sidebar.write(f"**India 10Y Bond Yield:** {INDIA_BOND_YIELD}%")
st.sidebar.info("This dashboard automates the screening process for the three most popular fundamental strategies in the Indian context.")
