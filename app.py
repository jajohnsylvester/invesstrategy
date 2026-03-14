import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
import requests
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Strategy Dashboard")

# --- SIDEBAR & CACHE CONTROL ---
st.sidebar.header("Global Settings")
if st.sidebar.button("♻️ Clear Cache & Refresh Data"):
    st.cache_data.clear()
    st.rerun()

bond_yield = st.sidebar.slider("India 10Y Bond Yield (%)", 4.0, 10.0, 6.68, 0.01)
target_count = st.sidebar.number_input("Target Stocks per Strategy", 1, 50, 10)

# --- DYNAMIC DATA ENGINE ---
@st.cache_data(ttl=86400)
def fetch_nifty_500_tickers():
    """Fetches the Nifty 500 list with browser-like headers to avoid 403 Errors."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
    except Exception as e:
        st.error(f"Error fetching Nifty 500: {e}")
        return ["TITAN.NS", "PIDILITIND.NS", "RELIANCE.NS", "HAL.NS", "CDSL.NS"]

def calculate_score(pe, roic, growth, ey, strategy):
    score = 0
    # Guard against 0 or None growth
    safe_growth = growth if growth and growth != 0 else 1 
    if strategy == "Coffee Can":
        score += (4 if roic > 20 else 2 if roic > 15 else 0)
        score += (4 if safe_growth > 15 else 2 if safe_growth > 10 else 0)
    elif strategy == "Magic Formula":
        score += (5 if roic > 25 else 2.5 if roic > 18 else 0)
        score += (5 if ey > 10 else 2.5 if ey > 6 else 0)
    elif strategy == "Lynch":
        peg = pe / safe_growth if safe_growth > 0 else 10
        score += (5 if peg < 1.0 else 2.5 if peg < 1.5 else 0)
        score += (5 if ey > bond_yield else 0)
    return round(score, 1)

@st.cache_data(ttl=3600)
def screen_universe(tickers, strategy, limit, b_yield):
    results = []
    # Progress bar for feedback
    progress_bar = st.progress(0, text=f"Scanning {strategy}...")
    
    for idx, ticker in enumerate(tickers[:150]): # Scan first 150 for speed
        if len(results) >= limit: break
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            # Use .get(key, 0) to avoid KeyErrors
            pe = info.get('trailingPE', 0)
            growth = info.get('earningsGrowth', 0)
            growth = (growth * 100) if growth is not None else 0
            roic = info.get('returnOnEquity', 0)
            roic = (roic * 100) if roic is not None else 0
            ey = (1 / pe * 100) if pe and pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0)
            rev_growth = (rev_growth * 100) if rev_growth is not None else 0

            # Strategy Logic
            match = False
            if strategy == "Coffee Can" and roic > 15 and rev_growth > 10: match = True
            elif strategy == "Magic Formula" and roic > 18 and ey > 6: match = True
            elif strategy == "Lynch" and (pe/growth if growth > 0 else 99) < 1.5 and ey > b_yield: match = True

            if match:
                results.append({
                    "Company": info.get('longName', ticker),
                    "Ticker": ticker,
                    "Sector": info.get('sector', 'Other'),
                    "F-Score": calculate_score(pe, roic, growth, ey, strategy),
                    "Price": info.get('currentPrice'),
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "EY %": round(ey, 2),
                    "PEG": round(pe/growth, 2) if growth > 0 else "N/A"
                })
        except: continue
        progress_bar.progress((idx + 1) / 150)
    
    progress_bar.empty()
    return pd.DataFrame(results)

# --- EXECUTION ---
ticker_list = fetch_nifty_500_tickers()
tabs = st.tabs(["Coffee Can", "Magic Formula", "Lynch (Market)"])

for i, strat in enumerate(["Coffee Can", "Magic Formula", "Lynch"]):
    with tabs[i]:
        df = screen_universe(ticker_list, strat, target_count, bond_yield)
        if not df.empty:
            st.dataframe(df.style.highlight_max(axis=0, subset=['F-Score']), use_container_width=True)
            st.plotly_chart(px.pie(df, names='Sector', hole=0.4), use_container_width=True)
        else:
            st.warning(f"No stocks currently match the {strat} filters in the first 150 tickers.")
