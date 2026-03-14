import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import io

# Set page config
st.set_page_config(page_title="John's Advanced Screener", layout="wide")

# --- DATA FETCHING & MANUAL CALCULATIONS ---

@st.cache_data
def fetch_nifty500_tickers():
    """Fetches the Nifty 500 list from NiftyIndices.com"""
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    return [f"{s}.NS" for s in df['Symbol'].tolist()]

def get_manual_metrics(ticker):
    """Calculates metrics manually to match Indian market standards."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials # Income Statement
        balance = stock.balance_sheet # Balance Sheet
        
        # 1. Price & Basic Info
        price = info.get('currentPrice', 0)
        eps = info.get('trailingEps', 1)
        bvps = info.get('bookValue', 1)
        
        # 2. Manual ROCE Calculation (Coffee Can Style)
        # EBIT / (Total Assets - Current Liabilities)
        ebit = financials.iloc[0, 0] if not financials.empty else info.get('ebitda', 0)
        assets = balance.loc['Total Assets'].iloc[0] if 'Total Assets' in balance.index else 1
        curr_liab = balance.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance.index else 0
        roce = (ebit / (assets - curr_liab)) * 100 if (assets - curr_liab) > 0 else 0

        # 3. Manual PEG Calculation (Peter Lynch Style)
        # We calculate 3-Year EPS Growth internally
        pe_ratio = price / eps if eps > 0 else 0
        eps_growth = 15 # Defaulting to a conservative 15% if history is unavailable
        if not financials.empty and financials.shape[1] >= 3:
            # Simple CAGR: ((Current EPS / EPS 3yrs ago)^(1/3)) - 1
            eps_current = financials.loc['Net Income'].iloc[0]
            eps_old = financials.loc['Net Income'].iloc[2]
            if eps_old > 0:
                eps_growth = ((eps_current / eps_old) ** (1/3) - 1) * 100
        peg = pe_ratio / eps_growth if eps_growth > 0 else 0
        
        # 4. Earnings Yield (Magic Formula)
        ev = info.get('enterpriseValue', 1)
        ey = (ebit / ev) * 100 if ev > 0 else 0

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Price": price,
            "ROCE (%)": round(roce, 2),
            "PEG": round(peg, 2),
            "Earnings Yield (%)": round(ey, 2),
            "Graham Number": round((22.5 * eps * bvps)**0.5, 2),
            "EPS Growth (%)": round(eps_growth, 2)
        }
    except Exception as e:
        return None

# --- APP UI ---

st.title("🏛️ NIFTY 500 Strategy Dashboard")
st.markdown("Automated Screening: **Coffee Can** | **Magic Formula** | **Peter Lynch**")

if st.button("🚀 Run Analysis on NIFTY 500"):
    tickers = fetch_nifty500_tickers()
    analysis_list = []
    
    # Using a smaller subset for demo speed; increase for full list
    progress_bar = st.progress(0)
    subset = tickers[:30] # Change to tickers[:] for full Nifty 500
    
    for i, t in enumerate(subset):
        res = get_manual_metrics(t)
        if res: analysis_list.append(res)
        progress_bar.progress((i + 1) / len(subset))
    
    df = pd.DataFrame(analysis_list)

    # UI TABS
    tab1, tab2, tab3 = st.tabs(["☕ Coffee Can", "🧙 Magic Formula", "📈 Peter Lynch"])

    with tab1:
        st.subheader("Coffee Can: Consistent Compounders")
        cc_df = df[df['ROCE (%)'] > 15].sort_values('ROCE (%)', ascending=False)
        st.dataframe(cc_df, use_container_width=True)

    with tab2:
        st.subheader("Magic Formula: Best Value & Quality")
        df['EY_Rank'] = df['Earnings Yield (%)'].rank(ascending=False)
        df['ROCE_Rank'] = df['ROCE (%)'].rank(ascending=False)
        df['Magic_Rank'] = df['EY_Rank'] + df['ROCE_Rank']
        mf_df = df.sort_values('Magic_Rank').drop(columns=['EY_Rank', 'ROCE_Rank'])
        st.dataframe(mf_df, use_container_width=True)

    with tab3:
        st.subheader("Peter Lynch: Growth at Reasonable Price")
        # PEG < 1 is the sweet spot
        lynch_df = df[(df['PEG'] > 0) & (df['PEG'] < 1.5)].sort_values('PEG')
        st.dataframe(lynch_df, use_container_width=True)

else:
    st.info("Click the button to fetch and analyze the Nifty 500.")
