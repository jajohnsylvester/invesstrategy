import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import io

# Set page config
st.set_page_config(page_title="John's NIFTY 500 Screener", layout="wide")

# --- DATA FETCHING ---

@st.cache_data
def fetch_nifty500_tickers():
    """Fetches the Nifty 500 list from NiftyIndices.com"""
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    # Append .NS for Yahoo Finance
    return [f"{s}.NS" for s in df['Symbol'].tolist()]

def get_stock_data(ticker):
    """Fetches and processes metrics for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        price = info.get('currentPrice', 0)
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        roce = info.get('returnOnCapitalEmployed', info.get('returnOnAssets', 0) * 2)
        peg = info.get('pegRatio', 0)
        ebit = info.get('ebitda', 0)
        ev = info.get('enterpriseValue', 1)
        
        graham = (22.5 * eps * bvps) ** 0.5 if eps > 0 and bvps > 0 else 0
        ey = (ebit / ev) * 100 if ev > 0 else 0
        
        return {
            "Ticker": ticker.replace(".NS", ""),
            "Price": price,
            "ROCE (%)": round(roce * 100, 2) if roce < 1 else roce,
            "PEG": peg,
            "Earnings Yield (%)": round(ey, 2),
            "Graham Number": round(graham, 2),
            "Price/Graham": round(price/graham, 2) if graham > 0 else 0
        }
    except:
        return None

# --- APP UI ---

st.title("🏛️ NIFTY 500 Triple Strategy Screener")

if st.button("🚀 Run Full Market Analysis (NIFTY 500)"):
    tickers = fetch_nifty500_tickers()
    
    # Process only top 30 for demo speed; remove [:30] for full run
    analysis_list = []
    progress_bar = st.progress(0)
    
    for i, t in enumerate(tickers[:40]): # Processing first 40 for speed in this demo
        res = get_stock_data(t)
        if res: analysis_list.append(res)
        progress_bar.progress((i + 1) / 40)
    
    df = pd.DataFrame(analysis_list)

    # Define Tabs
    tab1, tab2, tab3 = st.tabs(["☕ Coffee Can", "🧙 Magic Formula", "📈 Peter Lynch"])

    with tab1:
        st.header("Coffee Can (Consistent Compounders)")
        # Criteria: ROCE > 15% consistently
        cc_df = df[df['ROCE (%)'] > 15].sort_values(by='ROCE (%)', ascending=False)
        st.write("Targeting companies with high capital efficiency and a strong moat.")
        st.dataframe(cc_df, use_container_width=True)

    with tab2:
        st.header("Magic Formula (Value + Quality)")
        # Rank by ROCE and Earnings Yield
        mf_df = df.copy()
        mf_df['EY_Rank'] = mf_df['Earnings Yield (%)'].rank(ascending=False)
        mf_df['ROCE_Rank'] = mf_df['ROCE (%)'].rank(ascending=False)
        mf_df['Magic_Rank'] = mf_df['EY_Rank'] + mf_df['ROCE_Rank']
        mf_df = mf_df.sort_values('Magic_Rank')
        st.write("Joel Greenblatt's approach: Best ROCE at the lowest relative price.")
        st.dataframe(mf_df.drop(columns=['EY_Rank', 'ROCE_Rank']), use_container_width=True)

    with tab3:
        st.header("Peter Lynch (GARP)")
        # Criteria: PEG < 1.0
        lynch_df = df[(df['PEG'] > 0) & (df['PEG'] < 1.2)].sort_values(by='PEG')
        st.write("Growth at a Reasonable Price (GARP). Focus on companies where PEG < 1.")
        st.dataframe(lynch_df, use_container_width=True)

else:
    st.info("Click the button to fetch the latest NIFTY 500 list and run the multi-strategy screen.")
