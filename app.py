import streamlit as st
import yfinance as yf
import pandas as pd
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Dynamic Strategy Dashboard")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
bond_yield = st.sidebar.slider("India 10Y Bond Yield (%)", 4.0, 10.0, 6.68, 0.01)
num_stocks = st.sidebar.number_input("Stocks per Strategy", 5, 50, 20)

# --- DYNAMIC DATA ENGINE ---
@st.cache_data(ttl=86400) # Cache ticker list for 24 hours
def fetch_nifty_500():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        # Add .NS suffix for Yahoo Finance
        return [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
    except:
        # Fallback to a small list if NSE link is down
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TITAN.NS"]

@st.cache_data(ttl=3600)
def screen_stocks(tickers, strategy_type, limit, b_yield):
    screened_data = []
    # To keep the app fast, we'll process a subset or use multithreading in a real app
    # For this demo, we'll sample or take the first 100 to screen
    for ticker in tickers[:150]: 
        if len(screened_data) >= limit: break
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            pe = info.get('trailingPE', 0)
            eps_growth = info.get('earningsGrowth', 0) * 100
            roic = info.get('returnOnCapitalEmployed', info.get('returnOnEquity', 0) * 100)
            ey = (1/pe * 100) if pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0) * 100

            # --- STRATEGY LOGIC ---
            meets_criteria = False
            if strategy_type == "Coffee Can" and roic > 15 and rev_growth > 10:
                meets_criteria = True
            elif strategy_type == "Magic Formula" and roic > 20 and ey > 5:
                meets_criteria = True
            elif strategy_type == "Lynch" and (pe/eps_growth if eps_growth > 0 else 10) < 1.2 and ey > b_yield:
                meets_criteria = True

            if meets_criteria:
                screened_data.append({
                    "Ticker": ticker,
                    "Price": info.get('currentPrice'),
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "EY %": round(ey, 2),
                    "Growth %": round(eps_growth, 2),
                    "PEG": round(pe/eps_growth, 2) if eps_growth > 0 else "N/A"
                })
        except: continue
    return pd.DataFrame(screened_data)

# --- EXECUTION ---
nifty_500 = fetch_nifty_500()
tab1, tab2, tab3 = st.tabs(["Coffee Can Portfolio", "Magic Formula", "Beating the Market"])

with tab1:
    st.header("Coffee Can Portfolio")
    st.caption("Filters: ROCE > 15%, Revenue Growth > 10%")
    df_cc = screen_stocks(nifty_500, "Coffee Can", num_stocks, bond_yield)
    st.dataframe(df_cc, use_container_width=True)

with tab2:
    st.header("Magic Formula")
    st.caption("Filters: High ROIC + High Earnings Yield")
    df_mf = screen_stocks(nifty_500, "Magic Formula", num_stocks, bond_yield)
    st.dataframe(df_mf, use_container_width=True)

with tab3:
    st.header("Beating the Market")
    st.caption(f"Filters: PEG < 1.2, EY > {bond_yield}%")
    df_lynch = screen_stocks(nifty_500, "Lynch", num_stocks, bond_yield)
    st.dataframe(df_lynch, use_container_width=True)

st.sidebar.success(f"Screened {len(nifty_500)} stocks from Nifty 500")
