import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="AHR999 Index (Web 1.0)", layout="centered")

# --- 2. 注入 Web 1.0 复古 CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
        font-family: "Times New Roman", Times, serif;
    }
    h1, h2, h3 {
        color: #000080 !important;
        font-family: "Times New Roman", Times, serif !important;
        border-bottom: 2px solid #808080;
        padding-bottom: 5px;
    }
    .old-table {
        width: 100%;
        border-collapse: collapse;
        border: 1px solid black;
        margin-bottom: 20px;
    }
    .old-table th {
        background-color: #E0E0E0;
        border: 1px solid black;
        padding: 5px;
        text-align: left;
    }
    .old-table td {
        border: 1px solid black;
        padding: 5px;
    }
    .stButton>button {
        background-color: #D4D0C8;
        color: black;
        border: 2px outset white;
        border-radius: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 核心计算 ---
@st.cache_data(ttl=3600)
def get_data(ticker):
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Close']].copy().sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df = df[df['Close'] > 0]
        return df
    except Exception:
        return pd.DataFrame()

def calculate_ahr999(ticker, df):
    df['GeoMean'] = df['Close'].rolling(window=200).apply(lambda x: np.exp(np.mean(np.log(x))))
    start_date = df.index[0]
    df['Days'] = (df.index - start_date).days
    
    if ticker == "BTC-USD":
        genesis = pd.Timestamp("2009-01-03")
        days_btc = (df.index - genesis).days
        df['Predicted'] = 10 ** (2.68 + 0.000579 * days_btc)
        buy_line, sell_line = 0.45, 1.2
        note = "Standard BTC Parameters"
    else:
        x = df['Days'].values
        y = np.log10(df['Close'].values)
        slope, intercept, _, _, _ = linregress(x, y)
        df['Predicted'] = 10 ** (intercept + slope * df['Days'])
        df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
        valid_ahr = df['AHR999'].dropna()
        buy_line = np.percentile(valid_ahr, 10)
        sell_line = np.percentile(valid_ahr, 90)
        note = f"Dynamic Regression (Slope: {slope:.5f})"

    if'AHR999' not in df.columns:
        df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
        
    return df, buy_line, sell_line, note

# --- 4. 侧边栏 ---
st.sidebar.markdown("### Configuration")
ticker = st.sidebar.selectbox("Asset:", ["BTC-USD", "ETH-USD"])
if st.sidebar.button("Update Data"):
    st.cache_data.clear()

# --- 5. 主页面 ---
st.markdown("# AHR999 Index Monitor")
st.markdown(f"**Asset:** {ticker} | **Date:** {datetime.now().strftime('%Y-%m-%d')}")
st.markdown("<hr>", unsafe_allow_html=True)

with st.spinner("Fetching data from Yahoo Finance..."):
    df = get_data(ticker)
    if not df.empty and len(df) > 200:
        data, buy, sell, note = calculate_ahr999(ticker, df)
        last = data.iloc[-1]
        ahr_val = last['AHR999']
        
        status = "HOLD"
        color = "black"
        if ahr_val <= buy: status, color = "BUY (Undervalued)", "green"
        elif ahr_val >= sell: status, color = "SELL (Overvalued)", "red"

        st.markdown(f"""
        <table class="old-table">
            <tr><td><strong>Price</strong></td><td>${last['Close']:,.2f}</td></tr>
            <tr><td><strong>AHR999 Index</strong></td><td><b>{ahr_val:.4f}</b></td></tr>
            <tr><td><strong>Suggestion</strong></td><td style="color:{color}"><b>{status}</b></td></tr>
            <tr><td><strong>Algorithm</strong></td><td>{note}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        # 绘图
        st.markdown("### Historical Chart")
        plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman'], 
                             'axes.facecolor':'white', 'figure.facecolor':'white',
                             'grid.color':'#CCCCCC', 'grid.linestyle':':'})
        fig, ax = plt.subplots(figsize=(10, 5))
        subset = data.tail(365*4)
        ax.plot(subset.index, subset['AHR999'], color='#000080', label='Index')
        ax.axhline(buy, color='green', linestyle='--', label='Buy Line')
        ax.axhline(sell, color='red', linestyle='--', label='Sell Line')
        ax.legend()
        ax.grid(True)
        ax.set_title(f"{ticker} Analysis")
        st.pyplot(fig)
    else:
        st.error("Data Error.")
