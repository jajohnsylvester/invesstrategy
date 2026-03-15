import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import io
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="John's 2026 Strategy Dashboard", layout="wide")

# --- STRATEGY BENCHMARKS ---
STRATEGY_BENCHMARKS = {
    "TCS.NS": {"roce": 64.6, "peg": 2.1},
    "NESTLEIND.NS": {"roce": 95.6, "peg": 4.5},
    "BEL.NS": {"roce": 38.8, "peg": 0.95},
    "COALINDIA.NS": {"roce": 48.0, "ey": 16.5},
    "SHILCTECH.NS": {"roce": 71.3, "peg": 0.45}
}

@st.cache_data
def fetch_nifty500_tickers():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    return [f"{s}.NS" for s in df['Symbol'].tolist()]

def get_comprehensive_data(ticker, horizon_months):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Dynamic history based on horizon
        hist = stock.history(period=f"{horizon_months + 1}mo")
        
        price = info.get('currentPrice', 0)
        eps = info.get('trailingEps', 1)
        roce = info.get('returnOnCapitalEmployed', info.get('returnOnAssets', 0) * 2) * 100
        if ticker in STRATEGY_BENCHMARKS: roce = STRATEGY_BENCHMARKS[ticker]['roce']
        
        peg = info.get('pegRatio', (price/eps)/15)
        if ticker in STRATEGY_BENCHMARKS: peg = STRATEGY_BENCHMARKS[ticker].get('peg', peg)
        
        ebit = info.get('ebitda', 0)
        ev = info.get('enterpriseValue', 1)
        ey = (ebit / ev) * 100 if ev > 0 else 0
        
        # Momentum for selected horizon
        momentum = 0
        if len(hist) > 20:
            momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100

        # --- EXIT LOGIC ---
        exit_reason = ""
        is_exit = False
        
        # 1. Fundamental Exit (Coffee Can)
        if roce < 15:
            is_exit = True
            exit_reason = "ROCE < 15% (Moat Failure)"
        
        # 2. Valuation Exit (Lynch/Magic)
        if peg > 3.0:
            is_exit = True
            exit_reason = "PEG > 3.0 (Severe Overvaluation)"
            
        # 3. Momentum Exit (For 3-6 month horizons)
        if horizon_months <= 6 and momentum < -5:
            is_exit = True
            exit_reason = f"Trend Reversal ({horizon_months}m Mom < -5%)"

        signal = "🔴 EXIT" if is_exit else ("🟢 HOLD" if (roce > 20 and peg < 1.5) else "🟡 WATCH")

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Price": price,
            "ROCE (%)": round(roce, 2),
            "PEG": round(peg, 2),
            "EY (%)": round(ey, 2),
            "Momentum (%)": round(momentum, 2),
            "Acquirer Multiple": round(ev/ebit, 2) if ebit > 0 else 0,
            "Signal": signal,
            "Exit Reason": exit_reason if is_exit else "Fundamentals Intact"
        }
    except: return None

# --- UI ---
st.title("🏛️ Multi-Strategy NIFTY 500 Dashboard")

with st.sidebar:
    st.header("⚙️ Configuration")
    horizon = st.selectbox("Select Investment Horizon", [3, 6, 9], format_func=lambda x: f"{x} Months")
    st.divider()
    st.markdown("""
    **Exit Protocol:**
    - **3-6 Months:** Focus on Price Momentum. Exit if trend reverses.
    - **9 Months+:** Focus on ROCE. Exit if business efficiency drops.
    """)

# Legend
with st.expander("📖 View Strategy Rules & Exit Criteria"):
    c1, c2, c3 = st.columns(3)
    c1.success("**🟢 HOLD**: Strong Moat + Fair Valuation.")
    c2.warning("**🟡 WATCH**: Moderate risk; valuation creeping up.")
    c3.error("**🔴 EXIT**: Sell immediately due to Moat failure or extreme overvaluation.")

if st.button(f"🚀 Run {horizon}-Month Analysis"):
    all_t = fetch_nifty500_tickers()
    priority = [t for t in all_t if t in STRATEGY_BENCHMARKS.keys()]
    watchlist = priority + [t for t in all_t if t not in priority][:35]
    
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        data = get_comprehensive_data(t, horizon)
        if data: results.append(data)
        prog.progress((i + 1) / len(watchlist))
    
    df = pd.DataFrame(results)

    # Style function for Exit column
    def style_exit(v):
        return 'color: red; font-weight: bold' if v != "Fundamentals Intact" else 'color: gray'

    tabs = st.tabs(["☕ Coffee Can", "🧙 Magic Formula", "📈 Momentum", "💰 Acquirer", "🎯 Exit Report"])

    with tabs[0]:
        st.dataframe(df[df['ROCE (%)'] > 15].sort_values('ROCE (%)', ascending=False), use_container_width=True)
    with tabs[1]:
        df['Rank'] = df['ROCE (%)'].rank(ascending=False) + df['EY (%)'].rank(ascending=False)
        st.dataframe(df.sort_values('Rank'), use_container_width=True)
    with tabs[2]:
        st.dataframe(df.sort_values('Momentum (%)', ascending=False), use_container_width=True)
    with tabs[3]:
        st.dataframe(df.sort_values('Acquirer Multiple'), use_container_width=True)
    with tabs[4]:
        st.subheader("⚠️ Stocks Triggering Exit Signals")
        exit_df = df[df['Signal'] == "🔴 EXIT"][['Ticker', 'Price', 'Signal', 'Exit Reason']]
        st.dataframe(exit_df.style.applymap(lambda x: 'background-color: #ff4b4b', subset=['Signal']), use_container_width=True)
