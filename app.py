import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

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
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TITAN.NS"]

@st.cache_data(ttl=3600)
def screen_universe(tickers, strategy, limit, b_yield):
    results = []
    for ticker in tickers:
        if len(results) >= limit:
            break
        try:
            s = yf.Ticker(ticker)
            info = s.info
            
            pe = info.get('trailingPE', 0)
            eps_growth = info.get('earningsGrowth', 0) * 100
            # Use ROE as a reliable proxy for ROIC from info dictionary
            roic = info.get('returnOnEquity', 0) * 100 
            ey = (1 / pe * 100) if pe > 0 else 0
            rev_growth = info.get('revenueGrowth', 0) * 100

            match = False
            if strategy == "Coffee Can" and roic > 15 and rev_growth > 10:
                match = True
            elif strategy == "Magic Formula" and roic > 20 and ey > 6:
                match = True
            elif strategy == "Lynch" and (pe/eps_growth if eps_growth > 0 else 99) < 1.5 and ey > b_yield:
                match = True

            if match:
                results.append({
                    "Company Name": info.get('longName', ticker),
                    "Ticker": ticker,
                    "Sector": info.get('sector', 'Unknown'),
                    "Price": info.get('currentPrice'),
                    "P/E": round(pe, 2),
                    "ROIC %": round(roic, 2),
                    "EY %": round(ey, 2),
                    "Growth %": round(eps_growth, 2),
                    "PEG": round(pe/eps_growth, 2) if eps_growth > 0 else "N/A"
                })
        except:
            continue
    return pd.DataFrame(results)

def display_strategy_tab(strategy_name, description, sell_rule):
    st.header(strategy_name)
    st.info(f"**Strategy:** {description}")
    st.markdown(f"**Sell Rule:** {sell_rule}")
    
    df = screen_universe(ticker_list, strategy_name, target_count, bond_yield)
    
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Sector Breakdown Visualization
        st.markdown("### 🍩 Sector Distribution")
        fig = px.pie(df, names='Sector', title=f'Industry Exposure: {strategy_name}',
                     hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No stocks found matching these criteria currently.")

# --- EXECUTION ---
ticker_list = fetch_nifty_500_tickers()
tab1, tab2, tab3 = st.tabs(["Coffee Can Portfolio", "Magic Formula", "Beating the Market"])

with tab1:
    display_strategy_tab("Coffee Can", 
                         "Buy 'Consistent Compounders' (ROIC > 15%, Growth > 10%).",
                         "Only sell if management integrity is compromised.")

with tab2:
    display_strategy_tab("Magic Formula", 
                         "High ROIC (Quality) + High Earnings Yield (Value).",
                         "Strictly rebalance every 12 months.")

with tab3:
    display_strategy_tab("Lynch", 
                         "Growth at a Reasonable Price (PEG < 1.5, EY > Bond Yield).",
                         "Sell Stalwarts at 40% gain; Fast Growers if P/E > 2x Growth.")
