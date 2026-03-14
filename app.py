import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Strategy Dashboard")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
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
        return ["TITAN.NS", "PIDILITIND.NS", "ASIANPAINT.NS", "RELIANCE.NS", "TCS.NS", "HAL.NS", "CDSL.NS"]

def calculate_score(pe, roic, growth, ey, strategy):
    score = 0
    if strategy == "Coffee Can":
        score += (3 if roic > 20 else 1.5 if roic > 15 else 0)
        score += (3 if growth > 15 else 1.5 if growth > 10 else 0)
        score += (4 if pe < 40 else 2 if pe < 60 else 0)
    elif strategy == "Magic Formula":
        score += (5 if roic > 25 else 2.5 if roic > 18 else 0)
        score += (5 if ey > 10 else 2.5 if ey > 6 else 0)
    elif strategy == "Lynch":
        peg = pe / growth if growth > 0 else 10
        score += (4 if peg < 1.0 else 2 if peg < 1.5 else 0)
        score += (3 if ey > bond_yield else 0)
        score += (3 if growth > 20 else 1.5)
    return round(score, 1)

@st.cache_data(ttl=3600)
def screen_universe(tickers, strategy, limit, b_yield):
    results = []
    for ticker in tickers:
        if len(results) >= limit: break
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            # Fundamentals
            pe = info.get('trailingPE', 0)
            eps = info.get('trailingEps', 0)
            bvps = info.get('bookValue', 0)
            growth = info.get('earningsGrowth', 0) * 100
            roic = info.get('returnOnEquity', 0) * 100 
            ey = (1 / pe * 100) if pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0) * 100
            price = info.get('currentPrice')

            # --- VALUATION: GRAHAM NUMBER & SIGNAL ---
            # Graham Number = sqrt(22.5 * EPS * BVPS)
            graham_num = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
            margin = ((graham_num - price) / graham_num) * 100 if graham_num > 0 else -100
            
            signal = "HOLD"
            if price < graham_num:
                signal = "BUY (Under Valued)"
            elif price > (graham_num * 1.5):
                signal = "SELL (Over Valued)"

            # --- STRATEGY FILTERS ---
            match = False
            if strategy == "Coffee Can" and roic > 15 and rev_growth > 10: match = True
            elif strategy == "Magic Formula" and roic > 18 and ey > 6: match = True
            elif strategy == "Lynch" and (pe/growth if growth > 0 else 99) < 1.5 and ey > b_yield: match = True

            if match:
                results.append({
                    "Company Name": info.get('longName', ticker),
                    "Ticker": ticker,
                    "Signal": signal,
                    "F-Score": calculate_score(pe, roic, growth, ey, strategy),
                    "Price": price,
                    "Graham No.": round(graham_num, 2),
                    "Margin %": round(margin, 2),
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "PEG": round(pe/growth, 2) if growth > 0 else "N/A"
                })
        except: continue
    return pd.DataFrame(results)

# --- UI EXECUTION ---
ticker_list = fetch_nifty_500_tickers()
tabs = st.tabs(["Coffee Can Portfolio", "Magic Formula", "Beating the Market"])
strategies = [
    ("Coffee Can", {"Min ROCE": "15%", "Min Rev Growth": "10%", "Valuation": "Graham No."}),
    ("Magic Formula", {"Min ROIC": "18%", "Min Earn Yield": "6%", "Valuation": "Earnings Yield"}),
    ("Lynch", {"Max PEG": "1.5", "Yield Gap": f">{bond_yield}%", "Valuation": "Price/Growth"})
]

for i, (name, criteria) in enumerate(strategies):
    with tabs[i]:
        st.header(name)
        st.subheader("Filter Criteria")
        cols = st.columns(len(criteria))
        for j, (label, val) in enumerate(criteria.items()):
            cols[j].metric(label, val)
        
        st.markdown("---")
        df = screen_universe(ticker_list, name, target_count, bond_yield)
        if not df.empty:
            # Color coding signals for better visibility
            def color_signal(val):
                color = 'green' if 'BUY' in val else 'red' if 'SELL' in val else 'gray'
                return f'color: {color}; font-weight: bold'

            st.dataframe(df.style.applymap(color_signal, subset=['Signal']), use_container_width=True, hide_index=True)
            st.plotly_chart(px.pie(df, names='Sector' if 'Sector' in df else 'Signal', title=f'{name} Analysis Mix', hole=0.4), use_container_width=True)
        else:
            st.warning("Fetching real-time data from Nifty 500...")
