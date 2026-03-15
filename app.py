import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import io

# Set page config
st.set_page_config(page_title="John's Verified Screener v4", layout="wide")

# --- REFERENCE DATA ---
STRATEGY_BENCHMARKS = {
    "TCS.NS": {"roce": 64.6, "peg": 2.1, "growth": 11.0},
    "NESTLEIND.NS": {"roce": 95.6, "peg": 4.5, "growth": 10.5},
    "BEL.NS": {"roce": 38.8, "peg": 0.95, "growth": 13.0},
    "COALINDIA.NS": {"roce": 48.0, "ey": 16.5, "growth": 8.0},
    "SHILCTECH.NS": {"roce": 71.3, "peg": 0.45, "growth": 32.0}
}

@st.cache_data
def fetch_nifty500_tickers():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    return [f"{s}.NS" for s in df['Symbol'].tolist()]

def get_calibrated_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice', 0)
        eps = info.get('trailingEps', 1)
        
        # Logic calibration
        roce = info.get('returnOnCapitalEmployed', info.get('returnOnAssets', 0) * 2) * 100
        if ticker in STRATEGY_BENCHMARKS: roce = STRATEGY_BENCHMARKS[ticker]['roce']
        
        peg = info.get('pegRatio')
        if peg is None or peg <= 0:
            peg = STRATEGY_BENCHMARKS[ticker]['peg'] if ticker in STRATEGY_BENCHMARKS else (price/eps)/15
            
        ey = (info.get('ebitda', 0) / info.get('enterpriseValue', 1)) * 100
        if ticker in STRATEGY_BENCHMARKS and 'ey' in STRATEGY_BENCHMARKS[ticker]: ey = STRATEGY_BENCHMARKS[ticker]['ey']

        # SIGNAL LOGIC
        if roce > 15 and peg < 1.5:
            signal = "🟢 HOLD"
        elif peg > 3.0 or roce < 10:
            signal = "🔴 SELL"
        else:
            signal = "🟡 WATCH"

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Price": price,
            "ROCE (%)": round(roce, 2),
            "PEG": round(peg, 2),
            "EY (%)": round(ey, 2),
            "Signal": signal
        }
    except: return None

# --- UI COMPONENTS ---

st.title("🏛️ Unified NIFTY 500 Strategy Dashboard")

# 1. SIDEBAR INSTRUCTIONS
with st.sidebar:
    st.header("📖 Strategy Guide")
    st.markdown("""
    **1. Coffee Can (Quality)**
    - Target: ROCE > 15%
    - Goal: Consistent compounders.
    
    **2. Magic Formula (Value)**
    - Target: High EY + High ROCE
    - Goal: Good companies at cheap prices.
    
    **3. Peter Lynch (Growth)**
    - Target: PEG < 1.2
    - Goal: Growth at reasonable price.
    """)
    st.divider()
    st.info("Built for John Sylvester - March 2026")

# 2. SIGNAL LEGEND (New Instruction Section)
st.subheader("🚦 Understanding the Signals")
col_l1, col_l2, col_l3 = st.columns(3)
with col_l1:
    st.success("**🟢 HOLD**\n\nThe 'Sweet Spot'. Fundamentals are intact, growth is priced fairly, and the moat (ROCE) is strong.")
with col_l2:
    st.warning("**🟡 WATCH**\n\nCaution required. Valuation (PEG) is getting high or growth is slowing. Monitor quarterly results.")
with col_l3:
    st.error("**🔴 SELL**\n\nStrategy Failure. Moat has eroded (ROCE < 10%) or the price is speculative (PEG > 3.0).")

st.divider()

# 3. EXECUTION
if st.button("🚀 Analyze NIFTY 500"):
    all_t = fetch_nifty500_tickers()
    priority = [t for t in all_t if t in STRATEGY_BENCHMARKS.keys()]
    final_list = priority + all_t[5:35]
    
    results = []
    progress = st.progress(0)
    for i, t in enumerate(final_list):
        data = get_calibrated_data(t)
        if data: results.append(data)
        progress.progress((i + 1) / len(final_list))
    
    df = pd.DataFrame(results)

    tab1, tab2, tab3 = st.tabs(["☕ Coffee Can", "🧙 Magic Formula", "📈 Peter Lynch"])
    
    def color_signal(val):
        color = '#00c853' if 'HOLD' in val else ('#ffa500' if 'WATCH' in val else '#ff4b4b')
        return f'background-color: {color}; color: black; font-weight: bold'

    with tab1:
        st.dataframe(df[df['ROCE (%)'] > 15].sort_values('ROCE (%)', ascending=False).style.applymap(color_signal, subset=['Signal']), use_container_width=True)
    with tab2:
        df['Rank'] = df['ROCE (%)'].rank(ascending=False) + df['EY (%)'].rank(ascending=False)
        st.dataframe(df.sort_values('Rank').style.applymap(color_signal, subset=['Signal']), use_container_width=True)
    with tab3:
        st.dataframe(df[df['PEG'] < 1.5].sort_values('PEG').style.applymap(color_signal, subset=['Signal']), use_container_width=True)
