"""
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from itertools import product

# --- 1. DATA LOADING ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # yfinance sometimes returns MultiIndex columns; flatten if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# --- 2. STRATEGY LOGIC ---
def run_strategy(df, short_ma, long_ma, rsi_period, rsi_upper, stop_loss_pct):
    data = df.copy()
    
    # Indicators
    data['SMA_S'] = ta.sma(data['Close'], length=short_ma)
    data['SMA_L'] = ta.sma(data['Close'], length=long_ma)
    data['RSI'] = ta.rsi(data['Close'], length=rsi_period)
    
    # Signal Generation: MA Crossover + RSI Filter
    # Buy when Short MA > Long MA AND RSI is not overbought
    data['Signal'] = 0
    data.loc[(data['SMA_S'] > data['SMA_L']) & (data['RSI'] < rsi_upper), 'Signal'] = 1
    
    # Backtest logic with Stop Loss
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = 0.0
    
    in_position = False
    entry_price = 0
    
    for i in range(1, len(data)):
        # Entry
        if not in_position and data['Signal'].iloc[i-1] == 1:
            in_position = True
            entry_price = data['Close'].iloc[i]
            
        # Exit (MA Cross Under OR Stop Loss)
        elif in_position:
            current_price = data['Close'].iloc[i]
            price_change = (current_price - entry_price) / entry_price
            
            # Stop Loss Triggered
            if price_change <= -stop_loss_pct:
                data.at[data.index[i], 'Strategy_Returns'] = -stop_loss_pct
                in_position = False
            # Exit Signal (MA Cross Under)
            elif data['SMA_S'].iloc[i-1] < data['SMA_L'].iloc[i-1]:
                data.at[data.index[i], 'Strategy_Returns'] = price_change
                in_position = False
            else:
                # Still holding
                data.at[data.index[i], 'Strategy_Returns'] = data['Returns'].iloc[i]
                
    data['Cum_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    total_profit = data['Cum_Returns'].iloc[-1] - 1
    return data, total_profit

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Stock Strategy Optimizer", layout="wide")
st.title("📈 Indian Stock Strategy Backtester")

# Sidebar - Inputs
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter NSE Ticker (e.g., SBIN.NS, IOC.NS, ADANIPOWER.NS)", "SBIN.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Grid Search Parameters
st.sidebar.subheader("Grid Search Ranges")
short_range = st.sidebar.slider("Short MA Range", 5, 50, (10, 20))
long_range = st.sidebar.slider("Long MA Range", 50, 200, (50, 100))
sl_pct = st.sidebar.number_input("Stop Loss %", 1.0, 10.0, 3.0) / 100

if st.button("Run Optimization & Backtest"):
    raw_data = load_data(ticker_input, start_date, end_date)
    
    if raw_data.empty:
        st.error("No data found. Ensure the ticker symbol includes '.NS' for NSE.")
    else:
        # --- 4. GRID SEARCH ---
        st.subheader(f"Optimizing for {ticker_input}...")
        best_profit = -float('inf')
        best_params = {}
        
        # Testing combinations
        for s_ma, l_ma in product(range(short_range[0], short_range[1], 5), 
                                 range(long_range[0], long_range[1], 10)):
            _, profit = run_strategy(raw_data, s_ma, l_ma, 14, 70, sl_pct)
            if profit > best_profit:
                best_profit = profit
                best_params = {'short': s_ma, 'long': l_ma}
        
        # Run strategy with best parameters
        final_df, final_profit = run_strategy(raw_data, best_params['short'], best_params['long'], 14, 70, sl_pct)
        
        # --- 5. RESULTS & PLOT ---
        col1, col2 = st.columns(2)
        col1.metric("Best Total Profit", f"{final_profit*100:.2f}%")
        col2.write(f"Best Parameters: Short MA: {best_params['short']}, Long MA: {best_params['long']}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Close'], name="Close Price", opacity=0.5))
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df['SMA_S'], name=f"SMA {best_params['short']}"))
        fig.add_trace(go.Scatter(x=final_df.index, y=final_df['SMA_L'], name=f"SMA {best_params['long']}"))
        fig.update_layout(title=f"{ticker_input} Strategy Results", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_航空=True)
        
        st.write("### Cumulative Returns Curve")
        st.line_chart(final_df['Cum_Returns'])
"""
