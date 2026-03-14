import streamlit as st
import yfinance as yf
import pandas as pd
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Dynamic Strategy Dashboard")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
bond_yield = st.sidebar.slider("India 10Y Bond Yield (%)", 4.0, 10.0, 6.68, 0.01)
num_stocks = st.sidebar.number_input("Target Stocks per Strategy", 10, 50, 20)

# --- DYNAMIC DATA ENGINE ---
@st.cache_data(ttl=86400) # Cache ticker list for 24 hours
def fetch_nifty_500():
    # URL for Nifty 500 index constituents
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        return [f"{symbol}.NS" for symbol in df['Symbol'].tolist()]
    except:
        # Hardcoded fallback list if NSE link is down
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "TITAN.NS", "PIDILITIND.NS"]

@st.cache_data(ttl=3600)
def screen_stocks(tickers, strategy_type, limit, b_yield):
    screened_data = []
    # To keep the app responsive, we scan up to 250 stocks to find the matches
    for ticker in tickers[:250]: 
        if len(screened_data) >= limit: break
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            # Fundamentals
            pe = info.get('trailingPE', 0)
            eps_growth = info.get('earningsGrowth', 0) * 100
            roic = info.get('returnOnCapitalEmployed', info.get('returnOnEquity', 0) * 100)
            ey = (1/pe * 100) if pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0) * 100
            name = info.get('longName', ticker)

            # --- STRATEGY LOGIC ---
            meets_criteria = False
            if strategy_type == "Coffee Can" and roic > 15 and rev_growth > 8:
                meets_criteria = True
            elif strategy_type == "Magic Formula" and roic > 18 and ey > 5:
                meets_criteria = True
            elif strategy_type == "Lynch" and (pe/eps_growth if eps_growth > 0 else 10) < 1.5 and ey > b_yield:
                meets_criteria = True

            if meets_criteria:
                screened_data.append({
                    "Company Name": name,
                    "Ticker": ticker,
                    "Price": info.get('currentPrice'),
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "EY %": round(ey, 2),
                    "Growth %": round(eps_growth, 2),
                    "PEG": round(pe/eps_growth, 2) if eps_growth > 0 else "N/A"
                })
        except: continue
    return pd.DataFrame(screened_data)

# --- EXECUTION ---
nifty_500 = fetch_nifty_500()
tab1, tab2, tab3 = st.tabs(["Coffee Can Portfolio", "Magic Formula", "Beating the Market"])

with tab1:
    st.header("Coffee Can Portfolio")
    st.info("Filters: ROCE > 15%, Revenue Growth > 8%")
    df_cc = screen_stocks(nifty_500, "Coffee Can", num_stocks, bond_yield)
    st.dataframe(df_cc, use_container_width=True, hide_index=True)

with tab2:
    st.header("Magic Formula")
    st.info("Filters: High ROIC + High Earnings Yield (Cheap Price)")
    df_mf = screen_stocks(nifty_500, "Magic Formula", num_stocks, bond_yield)
    st.dataframe(df_mf, use_container_width=True, hide_index=True)

with tab3:
    st.header("Beating the Market")
    st.info(f"Filters: PEG < 1.5, Earnings Yield > {bond_yield}% (Bond Yield)")
    df_lynch = screen_stocks(nifty_500, "Lynch", num_stocks, bond_yield)
    st.dataframe(df_lynch, use_container_width=True, hide_index=True)

st.sidebar.success(f"Nifty 500 universe connected. Found {len(df_cc)} CC, {len(df_mf)} MF, and {len(df_lynch)} Lynch stocks.")
