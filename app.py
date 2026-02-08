import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="Market Cycle Monitor", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. 数据获取与处理 (保持之前的健壮性修复) ---
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
        # 尝试 Yahoo
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

    # 清洗
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

# --- 3. 绘图逻辑 (修复重点) ---
def create_chart(df_btc, df_eth):
    c_price, c_buy, c_acc, c_sell = "#000000", "#228b22", "#4682b4", "#b22222"
    
    def get_title(df, name):
        if df.empty: return f"{name}: No Data"
        return f"<b>{name}</b>: ${df['Close'].iloc[-1]:,.2f} | Index: {df['AHR999'].iloc[-1]:.4f}"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.03)

    # 数据源注入
    data_sources = []
    data_sources.append((df_btc if not df_btc.empty else pd.DataFrame({'Close':[],'Predicted':[],'AHR999':[]}), True, "BTC"))
    data_sources.append((df_eth if not df_eth.empty else pd.DataFrame({'Close':[],'Predicted':[],'AHR999':[]}), False, "ETH"))

    # 绘制线条
    for df, vis, name in data_sources:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color=c_price, width=1.5), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name="Model", line=dict(color="purple", width=1, dash='dash'), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AHR999'], name="Index", line=dict(color="#d35400", width=1.5), visible=vis), row=2, col=1)

    # 绘制背景区域
    zones = [(0.45, c_buy, "BUY"), (1.2, c_acc, "ACCUM"), (4.0, c_sell, "RISK")]
    for y_val, color, txt in zones:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=color, annotation_text=txt, annotation_font=dict(color=color))

    # --- 修复 1: 配置 RangeSelector 添加 1W, 2W ---
    # --- 修复 2: 设置 yaxis_fixedrange=False 解决直线问题 ---
    
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
        margin=dict(t=110, l=40, r=40, b=40), # 增加顶部边距防止遮挡
        title=dict(text=get_title(df_btc, "BTC-USD"), x=0, y=0.98),
        showlegend=False,
        
        # 关键修复：允许 Y 轴缩放
        yaxis=dict(fixedrange=False), 
        yaxis2=dict(fixedrange=False)
    )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                # 新增 1W 和 2W
                dict(count=1, label="1W", step="week", stepmode="backward"),
                dict(count=2, label="2W", step="week", stepmode="backward"),
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

    # 配置 Y 轴 (Log 模式)
    # autorange=True 让 Plotly 尝试自动调整，但 Log 轴在局部数据下可能仍需手动缩放
    fig.update_yaxes(type="log", title="USD (Log)", row=1, col=1, autorange=True, fixedrange=False)
    fig.update_yaxes(type="log", title="Index (Log)", row=2, col=1, autorange=True, fixedrange=False)
    
    return fig

# --- 4. 主程序 ---
st.title("Market Cycle Monitor")

with st.spinner("Syncing market data..."):
    btc_df = get_data("BTC-USD")
    eth_df = get_data("ETH-USD")

if not btc_df.empty:
    fig = create_chart(btc_df, eth_df)
    # config 中移除 staticPlot，并允许 scrollZoom 以便用户可以滚轮缩放修复视图
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})
else:
    st.error("Unable to load data.")
