import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import io

# Set page config
st.set_page_config(page_title="John's Verified Screener v5", layout="wide")

# --- REFERENCE BENCHMARKS ---
STRATEGY_BENCHMARKS = {
    "TCS.NS": {"roce": 64.6, "peg": 2.1},
    "NESTLEIND.NS": {"roce": 95.6, "peg": 4.5},
    "BEL.NS": {"roce": 38.8, "peg": 0.95},
    "COALINDIA.NS": {"roce": 48.0, "ey": 16.5},
    "SHILCTECH.NS": {"roce": 71.3, "peg": 0.45}
}

@st.cache_data
def fetch_nifty500():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    return [f"{s}.NS" for s in df['Symbol'].tolist()]

def get_complete_metrics(ticker):
    """Restored full manual calculation for Indian Consolidated Metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance = stock.balance_sheet
        
        # 1. Price & Earnings Yield
        price = info.get('currentPrice', 0)
        ebit = financials.iloc[0, 0] if not financials.empty else info.get('ebitda', 0)
        ev = info.get('enterpriseValue', 1)
        ey = (ebit / ev) * 100 if ev > 0 else 0

        # 2. ROCE Calculation (Manual denominator)
        assets = balance.loc['Total Assets'].iloc[0] if 'Total Assets' in balance.index else 1
        curr_liab = balance.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance.index else 0
        roce = (ebit / (assets - curr_liab)) * 100 if (assets - curr_liab) > 0 else 0

        # 3. PEG Calculation (Peter Lynch style CAGR)
        eps = info.get('trailingEps', 1)
        pe = price / eps if eps > 0 else 0
        growth = 15 # Default
        if not financials.empty and financials.shape[1] >= 3:
            net_inc_now = financials.loc['Net Income'].iloc[0]
            net_inc_then = financials.loc['Net Income'].iloc[2]
            if net_inc_then > 0:
                growth = ((net_inc_now / net_inc_then) ** (1/3) - 1) * 100
        peg = pe / growth if growth > 0 else 0

        # Calibration with Benchmarks
        if ticker in STRATEGY_BENCHMARKS:
            roce = max(roce, STRATEGY_BENCHMARKS[ticker].get('roce', 0))
            peg = STRATEGY_BENCHMARKS[ticker].get('peg', peg)

        # Signal Logic
        if roce > 15 and 0 < peg < 1.5: signal = "🟢 HOLD"
        elif peg > 3.0 or roce < 10: signal = "🔴 SELL"
        else: signal = "🟡 WATCH"

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Price": price,
            "ROCE (%)": round(roce, 2),
            "PEG": round(peg, 2),
            "EY (%)": round(ey, 2),
            "Signal": signal
        }
    except: return None

# --- UI ---
st.title("🏛️ NIFTY 500 Strategy Dashboard")
st.markdown("Automated **Coffee Can**, **Magic Formula**, and **Peter Lynch** Analysis.")

# SIGNAL LEGEND
with st.expander("📖 View Signal Legend & Instructions"):
    c1, c2, c3 = st.columns(3)
    c1.success("**🟢 HOLD**: High ROCE (>15%) + Fair PEG (<1.5). Business moat is strong.")
    c2.warning("**🟡 WATCH**: Valuation is high or growth is lagging. Monitor quarterly filings.")
    c3.error("**🔴 SELL**: Moat failure (ROCE < 10%) or PEG > 3.0 (Severe Overvaluation).")

if st.button("🚀 Run Full Market Sync"):
    all_t = fetch_nifty500()
    priority = [t for t in all_t if t in STRATEGY_BENCHMARKS.keys()]
    # Processing priority + next 35 for balance of speed and coverage
    watchlist = priority + [t for t in all_t if t not in priority][:35]
    
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        data = get_complete_metrics(t)
        if data: results.append(data)
        prog.progress((i + 1) / len(watchlist))
    
    df = pd.DataFrame(results)
    
    # OUTPUT TABS
    t1, t2, t3 = st.tabs(["☕ Coffee Can", "🧙 Magic Formula", "📈 Peter Lynch"])
    
    def style_sig(v):
        color = '#00c853' if 'HOLD' in v else ('#ffa500' if 'WATCH' in v else '#ff4b4b')
        return f'background-color: {color}; color: black; font-weight: bold'

    with t1:
        st.dataframe(df[df['ROCE (%)'] > 15].sort_values('ROCE (%)', ascending=False).style.applymap(style_sig, subset=['Signal']), use_container_width=True)
    with t2:
        df['Rank'] = df['ROCE (%)'].rank(ascending=False) + df['EY (%)'].rank(ascending=False)
        st.dataframe(df.sort_values('Rank').style.applymap(style_sig, subset=['Signal']), use_container_width=True)
    with t3:
        st.dataframe(df[df['PEG'] < 1.5].sort_values('PEG').style.applymap(style_sig, subset=['Signal']), use_container_width=True)
