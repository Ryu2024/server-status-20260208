import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- 1. 页面配置 ---
st.set_page_config(page_title="Market Cycle Monitor", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .modebar {display: none !important;}
</style>
""", unsafe_allow_html=True)

# --- 2. 数据获取 ---
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
            data = requests.get(url, timeout=10).json()
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

# --- 3. 绘图逻辑 ---
def create_chart(df_btc, df_eth):
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    def get_t(df, n):
        if df.empty: return f"{n}: No Data"
        return f"<b>{n}</b>: ${df['Close'].iloc[-1]:,.2f} | Index: {df['AHR999'].iloc[-1]:.4f}"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.03)

    # Traces
    for df, vis, name in [(df_btc, True, "BTC"), (df_eth, False, "ETH")]:
        d = df if not df.empty else pd.DataFrame({'Close':[],'Predicted':[],'AHR999':[]})
        fig.add_trace(go.Scatter(x=d.index, y=d['Close'], name="Price", line=dict(color=c_p, width=1.5), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['Predicted'], name="Model", line=dict(color="purple", width=1, dash='dash'), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['AHR999'], name="Index", line=dict(color="#d35400", width=1.5), visible=vis), row=2, col=1)

    # Zones
    for y, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_font=dict(color=c))

    # --- 核心修复：更新布局 ---
    fig.update_layout(
        dragmode=False, # 禁用鼠标抓取拖动
        hovermode="x unified",
        template="plotly_white",
        height=750,
        margin=dict(t=110, l=40, r=40, b=40),
        title=dict(text=get_t(df_btc, "BTC-USD"), x=0, y=0.98),
        showlegend=False,
        
        # 币种切换按钮
        updatemenus=[dict(
            type="buttons", direction="left", active=0, x=0, y=1.12,
            buttons=[
                dict(label="BTC", method="update", args=[{"visible": [True, True, True, False, False, False]}, {"title.text": get_t(df_btc, "BTC-USD")}]),
                dict(label="ETH", method="update", args=[{"visible": [False, False, False, True, True, True]}, {"title.text": get_t(df_eth, "ETH-USD")}])
            ]
        )]
    )

    # --- 核心修复：时间按钮 (注意：必须去掉 fixedrange=True) ---
    fig.update_xaxes(
        row=1, col=1,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=14, label="2W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL")
            ]),
            x=1, xanchor="right", y=1.12
        )
    )

    # --- 核心修复：Y轴自动缩放 ---
    fig.update_yaxes(type="log", title="USD", row=1, col=1, autorange=True, fixedrange=False)
    fig.update_yaxes(type="log", title="Index", row=2, col=1, autorange=True, fixedrange=False)
    
    return fig

# --- 4. 运行 ---
st.title("Market Cycle Monitor")
with st.spinner("Syncing..."):
    btc_df, eth_df = get_data("BTC-USD"), get_data("ETH-USD")

if not btc_df.empty:
    fig = create_chart(btc_df, eth_df)
    # config 中 scrollZoom 设为 False，彻底防止滚轮缩放干扰
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
else:
    st.error("Data Error")
