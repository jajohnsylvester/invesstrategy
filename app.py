import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import math
import requests
import io
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Strategy Dashboard")

# --- SIDEBAR ---
st.sidebar.header("Global Settings")
if st.sidebar.button("♻️ Clear Cache & Refresh All"):
    st.cache_data.clear()
    st.rerun()

bond_yield = st.sidebar.slider("India 10Y Bond Yield (%)", 4.0, 10.0, 6.68, 0.01)
target_count = st.sidebar.number_input("Target Stocks per Strategy", 5, 50, 20)

# --- DATA ENGINE ---
@st.cache_data(ttl=86400)
def fetch_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
    except:
        return ["TITAN.NS", "PIDILITIND.NS", "RELIANCE.NS", "HAL.NS", "CDSL.NS", "ITC.NS", "SBIN.NS"]

def calculate_score(pe, roic, growth, ey, strategy):
    score = 0
    if strategy == "Coffee Can":
        score += (4 if roic > 18 else 2 if roic > 12 else 0)
        score += (4 if growth > 12 else 2 if growth > 8 else 0)
        score += (2 if pe < 60 else 0)
    elif strategy == "Magic Formula":
        score += (5 if roic > 20 else 2.5 if roic > 15 else 0)
        score += (5 if ey > 8 else 2.5 if ey > 5 else 0)
    elif strategy == "Beating the Market":
        peg = pe / growth if growth > 0 else 10
        score += (5 if peg < 1.2 else 2.5 if peg < 1.8 else 0)
        score += (5 if ey > bond_yield else 0)
    return round(score, 1)

@st.cache_data(ttl=3600)
def process_strategy(tickers, strategy, limit, b_yield):
    results = []
    # Progress feedback for the user
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Scanning a larger portion of the Nifty 500
    scan_limit = 400 
    for idx, ticker in enumerate(tickers[:scan_limit]):
        if len(results) >= limit: break
        status_text.text(f"Scanning {ticker} for {strategy}...")
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            # Metrics Extraction with Safety
            pe = info.get('trailingPE', 0)
            eps = info.get('trailingEps', 0)
            bvps = info.get('bookValue', 0)
            growth = info.get('earningsGrowth', 0)
            growth = (growth * 100) if growth is not None else 0
            roic = info.get('returnOnEquity', 0)
            roic = (roic * 100) if roic is not None else 0
            ey = (1 / pe * 100) if pe and pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0)
            rev_growth = (rev_growth * 100) if rev_growth is not None else 0
            price = info.get('currentPrice', 0)
            
            # Display Only: Graham Number
            graham_num = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
            
            # --- BROADENED FILTERS ---
            match = False
            signal = "HOLD"
            
            if strategy == "Coffee Can":
                # Loosened from 15/10 to 12/8 to populate the list
                if roic > 12 and rev_growth > 8:
                    match = True
                    if roic > 20 and rev_growth > 15: signal = "BUY (Quality)"
                    if pe > 85: signal = "SELL (Overpriced)"

            elif strategy == "Magic Formula":
                # Loosened from 18/6 to 15/4
                if roic > 15 and ey > 4:
                    match = True
                    if ey > 8: signal = "BUY (Deep Value)"
                    if ey < 3: signal = "SELL (Yield Low)"

            elif strategy == "Beating the Market":
                # Loosened PEG from 1.5 to 2.0
                if (pe/growth if growth > 0 else 99) < 2.0 and ey > (b_yield - 1):
                    match = True
                    peg = (pe/growth if growth > 0 else 99)
                    if peg < 1.0 and ey > b_yield: signal = "BUY (GARP)"
                    if growth > 0 and pe > (2.5 * growth): signal = "SELL (Growth Bubble)"

            if match:
                results.append({
                    "Company Name": info.get('longName', ticker),
                    "Ticker": ticker,
                    "Sector": info.get('sector', 'Other'),
                    "Signal": signal,
                    "F-Score": calculate_score(pe, roic, growth, ey, strategy),
                    "Price": price,
                    "Graham Number": round(graham_num, 2),
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "Growth %": round(growth, 2),
                    "PEG": round(pe/growth, 2) if growth > 0 else "N/A"
                })
        except: continue
        progress_bar.progress((idx + 1) / scan_limit)
        
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results)

# --- UI STYLING ---
def style_rows(val):
    if 'BUY' in str(val): return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    if 'SELL' in str(val): return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
    return ''

# --- EXECUTION ---
ticker_list = fetch_tickers()
tabs = st.tabs(["Coffee Can", "Magic Formula", "Beating the Market"])

meta = {
    "Coffee Can": ["ROIC > 12%, Rev Growth > 8%", "Buy world-class compounders."],
    "Magic Formula": ["ROIC > 15%, EY > 4%", "Good companies at fair prices."],
    "Beating the Market": [f"PEG < 2.0, EY > {bond_yield-1}%", "Growth at a reasonable price."]
}

for i, name in enumerate(meta.keys()):
    with tabs[i]:
        st.header(name)
        st.info(f"**Filters:** {meta[name][0]} | **Goal:** {meta[name][1]}")
        
        df = process_strategy(ticker_list, name, target_count, bond_yield)
        
        if not df.empty:
            # Export
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 Export {name}", csv, f"{name.lower()}.csv", "text/csv")
            
            # Display
            st.dataframe(df.style.applymap(style_rows, subset=['Signal'])
                         .highlight_max(subset=['F-Score'], color='#fff3cd'), 
                         use_container_width=True, hide_index=True)
            
            st.plotly_chart(px.pie(df, names='Sector', title=f'{name} Sector Exposure', hole=0.4), use_container_width=True)
        else:
            st.warning(f"No stocks found in the Nifty 500 scan matching {name} criteria.")
