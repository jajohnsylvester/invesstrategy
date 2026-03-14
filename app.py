import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Strategy Dashboard")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
st.caption(f"Last Market Sync: {current_time} (March 2026)")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("Global Settings")
bond_yield = st.sidebar.slider("India 10Y Bond Yield (%)", 4.0, 10.0, 6.68, 0.01)
target_count = st.sidebar.number_input("Target Stocks per Strategy", 10, 50, 20)

# --- DYNAMIC DATA ENGINE ---
@st.cache_data(ttl=86400)
def fetch_nifty_500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        return [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
    except:
        # Fallback to high-conviction tickers if NSE is unreachable
        return ["TITAN.NS", "PIDILITIND.NS", "ASIANPAINT.NS", "RELIANCE.NS", "TCS.NS", "HAL.NS", "CDSL.NS"]

def calculate_score(pe, roic, growth, ey, strategy):
    score = 0
    if strategy == "Coffee Can":
        score += (4 if roic > 20 else 2 if roic > 15 else 0)
        score += (4 if growth > 15 else 2 if growth > 10 else 0)
        score += (2 if pe < 50 else 0)
    elif strategy == "Magic Formula":
        score += (5 if roic > 25 else 2.5 if roic > 18 else 0)
        score += (5 if ey > 10 else 2.5 if ey > 6 else 0)
    elif strategy == "Lynch":
        peg = pe / growth if growth > 0 else 10
        score += (5 if peg < 1.0 else 2.5 if peg < 1.5 else 0)
        score += (5 if ey > bond_yield else 0)
    return round(score, 1)

@st.cache_data(ttl=3600)
def screen_universe(tickers, strategy, limit, b_yield):
    results = []
    for ticker in tickers:
        if len(results) >= limit: break
        try:
            s = yf.Ticker(ticker)
            info = s.info
            pe = info.get('trailingPE', 0)
            growth = info.get('earningsGrowth', 0) * 100
            roic = info.get('returnOnEquity', 0) * 100 
            ey = (1 / pe * 100) if pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0) * 100
            price = info.get('currentPrice')

            # --- NEW SIGNAL LOGIC (STRATEGY-BASED) ---
            signal = "HOLD"
            if strategy == "Coffee Can" and roic > 20 and rev_growth > 12: signal = "BUY (High Quality)"
            elif strategy == "Magic Formula" and ey > 8: signal = "BUY (Value Play)"
            elif strategy == "Lynch" and (pe/growth if growth > 0 else 99) < 1.0: signal = "BUY (Underpriced Growth)"

            match = False
            if strategy == "Coffee Can" and roic > 15 and rev_growth > 10: match = True
            elif strategy == "Magic Formula" and roic > 18 and ey > 6: match = True
            elif strategy == "Lynch" and (pe/growth if growth > 0 else 99) < 1.5 and ey > b_yield: match = True

            if match:
                results.append({
                    "Company Name": info.get('longName', ticker),
                    "Ticker": ticker,
                    "Sector": info.get('sector', 'Other'),
                    "Signal": signal,
                    "F-Score": calculate_score(pe, roic, growth, ey, strategy),
                    "Price": price,
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "EY %": round(ey, 2),
                    "Growth %": round(growth, 2),
                    "PEG": round(pe/growth, 2) if growth > 0 else "N/A"
                })
        except: continue
    return pd.DataFrame(results)

# --- STYLING & CONVERSION ---
def style_signal(val):
    if 'BUY' in val: color = '#d4edda'; text = '#155724'
    else: color = 'transparent'; text = 'inherit'
    return f
