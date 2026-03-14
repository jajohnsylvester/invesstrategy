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

# --- SIDEBAR & GLOBAL SETTINGS ---
st.sidebar.header("Global Settings")
bond_yield = st.sidebar.slider("India 10Y Bond Yield (%)", 4.0, 10.0, 6.68, 0.01)
target_count = st.sidebar.number_input("Target Stocks per Strategy", 5, 50, 20)

if st.sidebar.button("♻️ Clear Cache & Refresh"):
    st.cache_data.clear()
    st.rerun()

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
        return ["TITAN.NS", "PIDILITIND.NS", "RELIANCE.NS", "HAL.NS", "CDSL.NS"]

def calculate_score(pe, roic, growth, ey, strategy):
    score = 0
    if strategy == "Coffee Can":
        score += (4 if roic > 20 else 2 if roic > 15 else 0)
        score += (4 if growth > 15 else 2 if growth > 10 else 0)
        score += (2 if pe < 50 else 0)
    elif strategy == "Magic Formula":
        score += (5 if roic > 25 else 2.5 if roic > 18 else 0)
        score += (5 if ey > 10 else 2.5 if ey > 6 else 0)
    elif strategy == "Beating the Market":
        peg = pe / growth if growth > 0 else 10
        score += (5 if peg < 1.0 else 2.5 if peg < 1.5 else 0)
        score += (5 if ey > bond_yield else 0)
    return round(score, 1)

@st.cache_data(ttl=3600)
def process_strategy(tickers, strategy, limit, b_yield):
    results = []
    progress_bar = st.progress(0, text=f"Scanning Nifty 500 for {strategy}...")
    
    # Increase scan range to 250 to ensure we find enough stocks
    for idx, ticker in enumerate(tickers[:250]):
        if len(results) >= limit: break
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            pe = info.get('trailingPE', 0)
            eps = info.get('trailingEps', 0)
            bvps = info.get('bookValue', 0)
            growth = info.get('earningsGrowth', 0) * 100
            roic = info.get('returnOnEquity', 0) * 100 
            ey = (1 / pe * 100) if pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0) * 100
            price = info.get('currentPrice', 0)

            # --- GRAHAM NUMBER (Display Only, not for Filtering) ---
            graham_num = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
            
            # --- STRATEGY FILTERS ---
            match = False
            signal = "HOLD"
            
            if strategy == "Coffee Can" and roic > 15 and rev_growth > 10:
                match = True
                if roic > 20 and rev_growth > 15: signal = "BUY (Quality)"
                
            elif strategy == "Magic Formula" and roic > 18 and ey > 6:
                match = True
                if ey > 10: signal = "BUY (Value)"
                
            elif strategy == "Beating the Market" and (pe/growth if growth > 0 else 99) < 1.5 and ey > b_yield:
                match = True
                if (pe/growth if growth > 0 else 99) < 1.0: signal = "BUY (Growth)"

            if match:
                results.append({
                    "Company Name": info.get('longName', ticker),
                    "Ticker": ticker,
                    "Sector": info.get('sector', 'Other'),
                    "Signal": signal,
                    "F-Score / 10": calculate_score(pe, roic, growth, ey, strategy),
                    "Price": price,
                    "Graham Number": round(graham_num, 2),
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "Growth %": round(growth, 2),
                    "PEG": round(pe/growth, 2) if growth > 0 else "N/A"
                })
        except: continue
        progress_bar.progress((idx + 1) / 250)
    progress_bar.empty()
    return pd.DataFrame(results)

# --- STYLING ---
def color_signal(val):
    color = '#d4edda' if 'BUY' in val else 'transparent'
    text = '#155724' if 'BUY' in val else 'inherit'
    return f'background-color: {color}; color: {text}; font-weight: bold;'

# --- MAIN UI ---
ticker_list = fetch_tickers()
tabs = st.tabs(["Coffee Can Portfolio", "Magic Formula", "Beating the Market"])

strategy_meta = {
    "Coffee Can": {
        "desc": "Focus on 'Consistent Compounders' with high capital efficiency.",
        "buy": "ROCE > 15%, Revenue Growth > 10% for 10 years.",
        "sell": "Only if management integrity fails or business model is disrupted.",
        "metrics": {"Min ROCE": "15%", "Min Growth": "10%"}
    },
    "Magic Formula": {
        "desc": "Buying 'Good Companies' at 'Cheap Prices'.",
        "buy": "High Return on Capital + High Earnings Yield.",
        "sell": "Strictly rebalance every 12 months.",
        "metrics": {"Min ROIC": "18%", "Min Earn Yield": "6%"}
    },
    "Beating the Market": {
        "desc": "Peter Lynch's GARP (Growth at a Reasonable Price).",
        "buy": f"PEG < 1.5 and Earnings Yield > {bond_yield}%.",
        "sell": "Stalwarts at 40% gain; Fast Growers if P/E > 2x Growth.",
        "metrics": {"Max PEG": "1.5", "Yield Gap": f">{bond_yield}%"}
    }
}

for i, (name, meta) in enumerate(strategy_meta.items()):
    with tabs[i]:
        st.header(name)
        st.info(f"**Strategy:** {meta['desc']}")
        
        # Display Guidelines
        col_left, col_right = st.columns(2)
        with col_left: st.success(f"**When to Buy:** {meta['buy']}")
        with col_right: st.warning(f"**When to Sell:** {meta['sell']}")
        
        # Display Metric Tiles
        st.markdown("### Filter Thresholds")
        m_cols = st.columns(len(meta['metrics']))
        for idx, (label, val) in enumerate(meta['metrics'].items()):
            m_cols[idx].metric(label, val)
            
        st.divider()
        
        # Process and Display Data
        df = process_strategy(ticker_list, name, target_count, bond_yield)
        
        if not df.empty:
            # Export CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 Export {name} (CSV)", csv, f"{name.lower().replace(' ','_')}.csv", "text/csv")
            
            # Display Table with Color Coding
            st.dataframe(df.style.applymap(color_signal, subset=['Signal'])
                         .highlight_max(subset=['F-Score / 10'], color='#fff3cd'), 
                         use_container_width=True, hide_index=True)
            
            # Sector Pie Chart
            st.plotly_chart(px.pie(df, names='Sector', title=f'Industry Exposure: {name}', hole=0.4), use_container_width=True)
        else:
            st.error("No matches found in the current scan range. Try lowering target count or refreshing.")
