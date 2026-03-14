import streamlit as st
import yfinance as yf
import pandas as pd
import math

# --- CONFIGURATION (MARCH 2026) ---
INDIA_BOND_YIELD = 6.68 

st.set_page_config(page_title="Strategy Dashboard", layout="wide")
st.title("📊 Strategy Dashboard")
st.markdown("---")

# --- TICKER LISTS (20 PER STRATEGY) ---
STRATEGY_DATA = {
    "Coffee Can Portfolio": [
        "TITAN.NS", "PIDILITIND.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "HDFCBANK.NS",
        "BAJFINANCE.NS", "DIVISLAB.NS", "TCS.NS", "HINDUNILVR.NS", "PAGEIND.NS",
        "BERGEPAINT.NS", "ASTRAL.NS", "RELAXO.NS", "ABBOTT.NS", "HDFCLIFE.NS",
        "LTIM.NS", "KOTAKBANK.NS", "DMART.NS", "LALPATHLAB.NS", "CHOLAFIN.NS"
    ],
    "Magic Formula": [
        "COALINDIA.NS", "ITC.NS", "HCLTECH.NS", "POWERGRID.NS", "NMDC.NS",
        "OFSS.NS", "CASTROLIND.NS", "BAJAJ-AUTO.NS", "RECLTD.NS", "PFC.NS",
        "INFY.NS", "TECHM.NS", "OIL.NS", "PETRONET.NS", "BEL.NS",
        "SUNTV.NS", "HGS.NS", "NATIONALUM.NS", "BAYERCROP.NS", "ZENSARTECH.NS"
    ],
    "Beating the Market": [
        "CDSL.NS", "HAL.NS", "MAZDOCK.NS", "VBL.NS", "RELIANCE.NS",
        "TATASTEEL.NS", "TRENT.NS", "KEI.NS", "POLYCAB.NS", "KPITTECH.NS",
        "FLUOROCHEM.NS", "CUMMINSIND.NS", "AIAENG.NS", "IGL.NS", "JYOTHYLAB.NS",
        "RADICO.NS", "CREDITACC.NS", "MEDANTA.NS", "PHOENIXLTD.NS", "BSE.NS"
    ]
}

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_metrics(tickers):
    results = []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            i = s.info
            pe = i.get('trailingPE', 0)
            growth = i.get('earningsGrowth', 0) * 100
            ey = (1/pe * 100) if pe > 0 else 0
            
            results.append({
                "Ticker": t,
                "Price": i.get('currentPrice', 0),
                "P/E": round(pe, 2),
                "ROIC %": round(i.get('returnOnEquity', 0) * 100, 2), # Using ROE as proxy
                "EY %": round(ey, 2),
                "Growth %": round(growth, 2),
                "PEG": round(pe/growth, 2) if growth > 0 else "N/A"
            })
        except: continue
    return pd.DataFrame(results)

# --- UI TABS ---
tabs = st.tabs(list(STRATEGY_DATA.keys()))

for i, (strat_name, tickers) in enumerate(STRATEGY_DATA.items()):
    with tabs[i]:
        st.header(strat_name)
        
        # Display Guidelines based on the Strategy
        if strat_name == "Coffee Can Portfolio":
            st.info("**Guidelines:** Buy 10-year consistent compounders. Sell only if management integrity fails.")
        elif strat_name == "Magic Formula":
