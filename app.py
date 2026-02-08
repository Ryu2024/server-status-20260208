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
st.set_page_config(page_title="Market Cycle Monitor Pro", layout="wide")

st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. 数据逻辑 ---
@st.cache_data(ttl=3600)
def get_data(ticker):
    """获取并计算指标 (保留原有逻辑)"""
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty or'Close' not in df.columns: raise ValueError
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('Close', axis=1, level=0, drop_level=True)
    except:
        coin = "bitcoin" if "BTC" in ticker else "ethereum"
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=max&interval=daily"
        data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        df = pd.DataFrame(data['prices'], columns=['ts', 'Close'])
        df['Date'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('Date')

    df = df.sort_index()
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[df['Close'] > 0]
    
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

# --- 3. 绘图逻辑 (核心修改：增加 RangeSelector) ---
def create_pro_chart(df_btc, df_eth):
    c_price, c_buy, c_acc, c_sell = "#000000", "#228b22", "#4682b4", "#b22222"
    
    def get_title_str(df, name):
        last_p, last_i = df['Close'].iloc[-1], df['AHR999'].iloc[-1]
        return f"<b>{name}</b>: ${last_p:,.2f}  |  <b>Index</b>: {last_i:.4f}"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)

    # 添加 Traces (0-2: BTC, 3-5: ETH)
    for i, (df, visible, name) in enumerate([(df_btc, True, "BTC"), (df_eth, False, "ETH")]):
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"{name} Price", line=dict(color=c_price, width=1.5), visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name=f"{name} Model", line=dict(color="purple", width=1, dash='dash'), visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AHR999'], name=f"{name} Index", line=dict(color="#d35400", width=1.5), visible=visible), row=2, col=1)

    # 绘制阈值区域
    for y_val, label, color in [(0.45, "BUY ZONE", c_buy), (1.2, "ACCUMULATE", c_acc), (4.0, "HIGH RISK", c_sell)]:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=f"<b>{label}</b>", annotation_position="bottom right", annotation_font=dict(color=color, size=10))

    # --- 配置 X 轴范围选择器 (RangeSelector) ---
    fig.update_xaxes(
        row=2, col=1,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL")
            ]),
            bgcolor="#eeeeee",
            activecolor="#cccccc",
            x=1, xanchor="right", y=1.1 # 放在图表中间右侧
        )
    )

    # --- 配置 BTC/ETH 切换按钮 ---
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", active=0, x=0, y=1.15,
            buttons=[
                dict(label="BTC-USD", method="update", args=[{"visible": [True, True, True, False, False, False]}, {"title.text": get_title_str(df_btc, "BTC-USD")}]),
                dict(label="ETH-USD", method="update", args=[{"visible": [False, False, False, True, True, True]}, {"title.text": get_title_str(df_eth, "ETH-USD")}])
            ]
        )]
    )

    # 样式美化
    fig.update_yaxes(type="log", row=1, col=1, title="USD", gridcolor="#f0f0f0")
    fig.update_yaxes(type="log", row=2, col=1, title="Index", gridcolor="#f0f0f0")
    fig.update_layout(
        title=dict(text=get_title_str(df_btc, "BTC-USD"), font=dict(size=18), x=0, y=0.97),
        template="plotly_white", height=800,
        margin=dict(t=120, l=50, r=50, b=50),
        hovermode="x unified",
        showlegend=False
    )
    return fig

# --- 4. 运行逻辑 ---
with st.spinner("Syncing data..."):
    btc_data = get_data("BTC-USD")
    eth_data = get_data("ETH-USD")

if not btc_data.empty and not eth_data.empty:
    fig = create_pro_chart(btc_data, eth_data)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.error("Data connection lost.")
