import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 页面配置 ---
st.set_page_config(page_title="Market Cycle Monitor", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 数据获取 (包含防错机制) ---
def fetch_coingecko(ticker):
    try:
        coin = "bitcoin" if "BTC" in ticker else "ethereum"
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=max&interval=daily"
        data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10).json()
        df = pd.DataFrame(data['prices'], columns=['ts', 'Close'])
        df['Date'] = pd.to_datetime(df['ts'], unit='ms')
        return df.set_index('Date')[['Close']]
    except: 
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_data(ticker):
    df = pd.DataFrame()
    try:
        raw = yf.download(ticker, period="max", interval="1d", progress=False)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                try: df = raw.xs('Close', axis=1, level=0, drop_level=True)
                except KeyError: df = raw.iloc[:, 0].to_frame('Close')
            else:
                df = raw.copy()
            if'Close' not in df.columns: df = df.iloc[:, 0].to_frame('Close')
    except: pass

    if df.empty: df = fetch_coingecko(ticker)
    if df.empty: return df

    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[pd.to_numeric(df['Close'], errors='coerce') > 0]
    
    # 指标计算
    df['GeoMean'] = np.exp(np.log(df['Close']).rolling(200).mean())
    df['Days'] = (df.index - pd.Timestamp("2009-01-03")).days
    df = df[df['Days'] > 0].dropna()
    
    if "BTC" in ticker:
        df['Predicted'] = 10 ** (5.84 * np.log10(df['Days']) - 17.01)
    else:
        slope, intercept, _, _, _ = linregress(np.log10(df['Days']), np.log10(df['Close']))
        df['Predicted'] = 10 ** (intercept + slope * np.log10(df['Days']))

    df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
    return df

# --- 绘图逻辑 (修复 step 参数) ---
def create_chart(df_btc, df_eth):
    c_price, c_buy, c_acc, c_sell = "#000000", "#228b22", "#4682b4", "#b22222"
    
    def get_title(df, name):
        if df.empty: return f"{name}: No Data"
        return f"<b>{name}</b>: ${df['Close'].iloc[-1]:,.2f} | Index: {df['AHR999'].iloc[-1]:.4f}"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.03)

    # 注入数据
    datas = []
    datas.append((df_btc if not df_btc.empty else pd.DataFrame({'Close':[],'Predicted':[],'AHR999':[]}), True, "BTC"))
    datas.append((df_eth if not df_eth.empty else pd.DataFrame({'Close':[],'Predicted':[],'AHR999':[]}), False, "ETH"))

    # 绘制
    for df, vis, name in datas:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color=c_price, width=1.5), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name="Model", line=dict(color="purple", width=1, dash='dash'), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AHR999'], name="Index", line=dict(color="#d35400", width=1.5), visible=vis), row=2, col=1)

    # 区域线
    zones = [(0.45, c_buy, "BUY"), (1.2, c_acc, "ACCUM"), (4.0, c_sell, "RISK")]
    for y_val, color, txt in zones:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=color, annotation_text=txt, annotation_font=dict(color=color))

    # 布局配置
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", active=0, x=0, y=1.12,
            buttons=[
                dict(label="BTC", method="update", args=[{"visible": [True, True, True, False, False, False]}, {"title.text": get_title(df_btc, "BTC-USD")}]),
                dict(label="ETH", method="update", args=[{"visible": [False, False, False, True, True, True]}, {"title.text": get_title(df_eth, "ETH-USD")}])
            ]
        )],
        hovermode="x unified",
        template="plotly_white",
        height=750,
        margin=dict(t=110, l=40, r=40, b=40),
        title=dict(text=get_title(df_btc, "BTC-USD"), x=0, y=0.98),
        showlegend=False,
        
        # 允许鼠标缩放Y轴 (关键)
        yaxis=dict(fixedrange=False, autorange=True), 
        yaxis2=dict(fixedrange=False, autorange=True)
    )

    # 修复 range selector：使用 day 替代 week
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1W", step="day", stepmode="backward"),   # 修正点：7天
                dict(count=14, label="2W", step="day", stepmode="backward"),  # 修正点：14天
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL")
            ]),
            x=1, xanchor="right", y=1.12,
            font=dict(size=11)
        ),
        row=1, col=1
    )

    fig.update_yaxes(type="log", title="USD (Log)", row=1, col=1)
    fig.update_yaxes(type="log", title="Index (Log)", row=2, col=1)
    
    return fig

# --- 主程序 ---
st.title("Market Cycle Monitor")

with st.spinner("Syncing data..."):
    btc = get_data("BTC-USD")
    eth = get_data("ETH-USD")

if not btc.empty:
    fig = create_chart(btc, eth)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})
else:
    st.error("Data Load Error")
