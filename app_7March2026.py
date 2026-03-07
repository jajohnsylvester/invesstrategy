import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import quantstats as qs
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(page_title="Indian Stock Architect", layout="wide")

def calculate_capm(stock_ret, bench_ret):
    """Calculates Alpha and Beta using Linear Regression."""
    df = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    df.columns = ['Stock', 'Bench']
    X = df['Bench'].values.reshape(-1, 1)
    y = df['Stock'].values
    model = LinearRegression().fit(X, y)
    beta = model.coef_[0]
    alpha = (model.intercept_ * 252) * 100  # Annualized Alpha
    return round(alpha, 2), round(beta, 2)

def get_graham_number(info):
    """Calculates the intrinsic Graham Number."""
    eps = info.get('forwardEps', 0)
    bvps = info.get('bookValue', 0)
    if eps > 0 and bvps > 0:
        return round(np.sqrt(22.5 * eps * bvps), 2)
    return 0

# --- Sidebar Configuration ---
st.sidebar.title("🏗️ Portfolio Architect")
ticker = st.sidebar.text_input("Enter NSE Ticker", "TCS.NS")
benchmark = "^NSEI" # Nifty 50

# --- Data Fetching ---
try:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    st.title(f"Analysis: {info.get('longName', ticker)}")
    
    tab1, tab2 = st.tabs(["💎 Fundamentals & Strategy", "📊 Risk & Backtest"])

    with tab1:
        # 1. Key Metrics Row
        g_num = get_graham_number(info)
        curr_price = info.get('currentPrice', 1)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Current Price", f"₹{curr_price}")
        m_col2.metric("Graham Number", f"₹{g_num}", 
                     delta=f"{round(((g_num-curr_price)/curr_price)*100, 2)}% Margin")
        m_col3.metric("Piotroski Score", info.get('overallRisk', 'N/A')) # Proxy for screening
        m_col4.metric("Debt/Equity", round(info.get('debtToEquity', 0)/100, 2))

        # 2. Coffee Can & Value Analysis
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("☕ Coffee Can Filter")
            roce = info.get('returnOnCapitalEmployed', info.get('returnOnEquity', 0) * 1.2) * 100
            rev_growth = info.get('revenueGrowth', 0) * 100
            
            st.write(f"- **10Y Efficiency Proxy (ROE):** {round(roce, 2)}% {'✅' if roce > 15 else '❌'}")
            st.write(f"- **Revenue Growth:** {round(rev_growth, 2)}% {'✅' if rev_growth > 10 else '❌'}")
            st.write(f"- **Debt/Equity:** {round(info.get('debtToEquity', 0)/100, 2)} {'✅' if info.get('debtToEquity', 0) < 50 else '❌'}")
        
        with c2:
            st.subheader("📉 Value Analysis")
            is_undervalued = curr_price < g_num
            st.write(f"- **Graham Margin:** {'✅ Undervalued' if is_undervalued else '⚠️ Trading at Premium'}")
            st.write(f"- **PEG Ratio:** {info.get('pegRatio', 'N/A')} {'✅' if info.get('pegRatio', 0) < 1.2 else '❌'}")

    with tab2:
        if st.button("🚀 Run Full Backtest (5 Years)"):
            hist = yf.download(ticker, period="5y")['Close'].tz_localize(None)
            bench = yf.download(benchmark, period="5y")['Close'].tz_localize(None)
            
            s_ret = hist.pct_change().dropna()
            b_ret = bench.pct_change().dropna()
            
            alpha, beta = calculate_capm(s_ret, b_ret)
            
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_col1.metric("Annualized Alpha", f"{alpha}%")
            r_col2.metric("Beta", beta)
            r_col3.metric("Sharpe Ratio", round(qs.stats.sharpe(s_ret), 2))
            r_col4.metric("Sortino Ratio", round(qs.stats.sortino(s_ret), 2))
            
            st.write("### Cumulative Returns vs Nifty 50")
            fig = qs.plots.returns(s_ret, b_ret, show=False)
            st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading {ticker}. Please ensure it follows the format 'SYMBOL.NS'.")





