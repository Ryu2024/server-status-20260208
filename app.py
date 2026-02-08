import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 页面配置 ---
st.set_page_config(page_title="Market Cycle Monitor", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .modebar { display: none !important; } /* 隐藏右上角工具栏 */
</style>
""", unsafe_allow_html=True)

# --- 2. 数据获取 (保持原逻辑) ---
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

    if df.empty:
        try:
            coin = "bitcoin" if "BTC" in ticker else "ethereum"
            url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=max&interval=daily"
            data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
            df = pd.DataFrame(data['prices'], columns=['ts', 'Close'])
            df['Date'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.set_index('Date')[['Close']]
        except: return pd.DataFrame()
    
    if df.empty: return df

    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[pd.to_numeric(df['Close'], errors='coerce') > 0]
    
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

# --- 3. 绘图逻辑 (深度改造) ---
def create_chart(df_btc, df_eth):
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    # 1. 创建子图 (开启 shared_xaxes)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)

    # --- 绘制 BTC (Row 1 & Row 2) ---
    if not df_btc.empty:
        # Price (y)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name="BTC Price", line=dict(color=c_p, width=1.5), visible=True), row=1, col=1)
        # Model (y)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Predicted'], name="BTC Model", line=dict(color="purple", width=1, dash='dash'), visible=True), row=1, col=1)
        # Index (y2)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['AHR999'], name="BTC Index", line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # --- 绘制 ETH (使用影子轴 y3, y4) ---
    if not df_eth.empty:
        # Price (y3 - Shadow)
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Close'], name="ETH Price", line=dict(color=c_p, width=1.5), yaxis="y3", visible=False)) 
        # Model (y3 - Shadow)
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Predicted'], name="ETH Model", line=dict(color="purple", width=1, dash='dash'), yaxis="y3", visible=False))
        # Index (y4 - Shadow)
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['AHR999'], name="ETH Index", line=dict(color="#d35400", width=1.5), yaxis="y4", visible=False))

    # --- 背景区域 (Row 2) ---
    for y_val, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_position="top left", annotation_font=dict(color=c, size=10))

    # --- 按钮定义 (BTC/ETH 切换) ---
    # 这里的关键是同时切换 traces 的 visible 和 y轴 的 visible
    
    # 准备标题
    t_btc = f"<b>BTC-USD</b>: ${df_btc['Close'].iloc[-1]:,.2f}" if not df_btc.empty else "BTC"
    t_eth = f"<b>ETH-USD</b>: ${df_eth['Close'].iloc[-1]:,.2f}" if not df_eth.empty else "ETH"

    btn_btc = dict(
        label="BTC", method="update",
        args=[
            {"visible": [True, True, True, False, False, False]}, # 显示前3个，隐藏后3个
            {
                "title.text": t_btc,
                "yaxis.visible": True, "yaxis2.visible": True,    # 显示 BTC 轴
                "yaxis3.visible": False, "yaxis4.visible": False  # 隐藏 ETH 轴
            }
        ]
    )
    
    btn_eth = dict(
        label="ETH", method="update",
        args=[
            {"visible": [False, False, False, True, True, True]}, # 隐藏前3个，显示后3个
            {
                "title.text": t_eth,
                "yaxis.visible": False, "yaxis2.visible": False,  # 隐藏 BTC 轴
                "yaxis3.visible": True, "yaxis4.visible": True    # 显示 ETH 轴
            }
        ]
    )

    # --- 布局配置 (核心修复点) ---
    
    # 1. 全局清除所有 X 轴上的 rangeselector (按钮) 和 rangeslider (防止重复)
    fig.update_xaxes(rangeselector=dict(visible=False), rangeslider=dict(visible=False), fixedrange=False)

    # 2. 仅在最底部 (Row 2) 开启 Range Slider
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05, bgcolor="#f4f4f4"),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=80, l=40, r=40, b=60),
        title=dict(text=t_btc, x=0.01, y=0.96),
        hovermode="x unified",
        showlegend=False,
        dragmode="pan", # 鼠标默认为平移模式
        
        # 按钮组位置
        updatemenus=[dict(
            type="buttons", direction="left", active=0, x=0.01, y=1.08,
            buttons=[btn_btc, btn_eth],
            bgcolor="white", bordercolor="#e0e0e0", borderwidth=1
        )],
        
        # --- Y轴定义 ---
        # BTC Axes
        yaxis=dict(domain=[0.35, 1], type="log", title="Price", fixedrange=False),
        yaxis2=dict(domain=[0, 0.30], type="log", title="Index", fixedrange=False),
        
        # ETH Axes (Shadow - 默认隐藏)
        # overlaying="y" 意味着它和 y 轴共用同一个区域
        yaxis3=dict(domain=[0.35, 1], anchor="x", overlaying="y", side="left", type="log", visible=False, showgrid=False, fixedrange=False),
        yaxis4=dict(domain=[0, 0.30], anchor="x", overlaying="y2", side="left", type="log", visible=False, showgrid=False, fixedrange=False)
    )

    return fig

# --- 4. 主程序 ---
st.title("Market Cycle Monitor")
with st.spinner("Syncing data..."):
    btc_df, eth_df = get_data("BTC-USD"), get_data("ETH-USD")

if not btc_df.empty:
    # 直接传入全量数据，不需要 Pandas 切片，交给 Plotly 的 Slider 处理
    fig = create_chart(btc_df, eth_df)
    
    st.plotly_chart(fig, use_container_width=True, 
                    config={
                        'displayModeBar': False, # 隐藏悬浮工具栏
                        'scrollZoom': True,      # 允许滚轮缩放
                        'doubleClick': 'reset',  # 双击重置
                        'showAxisRangeEntryBoxes': False
                    })
else:
    st.error("Data Load Failed")
