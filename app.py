import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import io

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
        
        # Momentum
        momentum = 0
        if len(hist) > 20:
            momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100

        # --- CONSOLIDATED SIGNAL LOGIC ---
        action = "🟡 WATCH"
        reason = "Fundamentals Neutral"

        # 1. EXIT LOGIC (Priority)
        if roce < 15 or peg > 3.0 or (horizon_months <= 6 and momentum < -10):
            action = "🔴 EXIT"
            reason = "Moat failure or Overvalued/Trend break"
        
        # 2. BUY LOGIC (Strict Criteria)
        elif roce > 25 and 0.1 < peg < 1.0 and ey > 8:
            action = "🔵 BUY"
            reason = "Undervalued Quality (High ROCE + Low PEG)"

        # 3. HOLD LOGIC
        elif roce > 18 and peg < 1.8:
            action = "🟢 HOLD"
            reason = "Steady Compounder"

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Price": price,
            "ROCE (%)": round(roce, 2),
            "PEG": round(peg, 2),
            "EY (%)": round(ey, 2),
            "Momentum (%)": round(momentum, 2),
            "Signal": action,
            "Action Detail": reason
        }
    except: return None

# --- UI ---
st.title("🏛️ Multi-Strategy NIFTY 500 Dashboard")

with st.sidebar:
    st.header("⚙️ Configuration")
    horizon = st.selectbox("Investment Horizon", [3, 6, 9], format_func=lambda x: f"{x} Months")
    st.divider()
    st.markdown("""
    **Signal Glossary:**
    - **🔵 BUY:** High Quality (ROCE > 25%) + Cheap (PEG < 1.0).
    - **🟢 HOLD:** Core position; fundamentals are stable.
    - **🔴 EXIT:** Fundamental breakdown or extreme price risk.
    """)

if st.button(f"🚀 Analyze Market ({horizon}m Horizon)"):
    all_t = fetch_nifty500_tickers()
    priority = [t for t in all_t if t in STRATEGY_BENCHMARKS.keys()]
    watchlist = priority + [t for t in all_t if t not in priority][:40]
    
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        data = get_comprehensive_data(t, horizon)
        if data: results.append(data)
        prog.progress((i + 1) / len(watchlist))
    
    df = pd.DataFrame(results)

    def style_signal(v):
        colors = {'🔵 BUY': '#1f77b4', '🟢 HOLD': '#2ca02c', '🔴 EXIT': '#d62728', '🟡 WATCH': '#ff7f0e'}
        return f'background-color: {colors.get(v, "white")}; color: white; font-weight: bold'

    tabs = st.tabs(["📊 Market Overview", "🔵 BUY List", "🔴 EXIT List", "🧙 Strategy Rankings"])

    with tabs[0]:
        st.subheader("Full Watchlist Analysis")
        st.dataframe(df.style.applymap(style_signal, subset=['Signal']), use_container_width=True)

    with tabs[1]:
        st.subheader("🔥 Fresh BUY Opportunities")
        st.write("Stocks meeting strict Quality + Value criteria.")
        st.dataframe(df[df['Signal'] == "🔵 BUY"].sort_values('ROCE (%)', ascending=False), use_container_width=True)

    with tabs[2]:
        st.subheader("⚠️ Critical EXIT Alerts")
        st.dataframe(df[df['Signal'] == "🔴 EXIT"], use_container_width=True)

    with tabs[3]:
        st.subheader("Top Ranks (Magic Formula & Momentum)")
        df['MF_Rank'] = df['ROCE (%)'].rank(ascending=False) + df['EY (%)'].rank(ascending=False)
        st.dataframe(df.sort_values('MF_Rank').head(15), use_container_width=True)
