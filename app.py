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

# --- 2. 样式定制 (CSS) - 移除了原有的按钮样式 ---
st.markdown("""
<style>
    /* 隐藏默认头部 */
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* 滑块样式微调 */
    div[data-testid="stSlider"] {
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 数据逻辑 ---
def fetch_coingecko(ticker):
    """备用数据源"""
    try:
        coin = "bitcoin" if "BTC" in ticker else "ethereum"
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=max&interval=daily"
        data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        df = pd.DataFrame(data['prices'], columns=['ts', 'Close'])
        df['Date'] = pd.to_datetime(df['ts'], unit='ms')
        return df.set_index('Date')[['Close']]
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_data(ticker):
    """获取并计算指标"""
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty: raise ValueError
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('Close', axis=1, level=0, drop_level=True)
        if'Close' not in df.columns: df = df.iloc[:, 0].to_frame('Close')
    except: 
        df = fetch_coingecko(ticker)
    
    if df.empty: return df

    # 清洗
    df = df.sort_index()
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[df['Close'] > 0]
    
    # 200日几何平均
    df['GeoMean'] = np.exp(np.log(df['Close']).rolling(200).mean())
    
    # 回归模型
    df['Days'] = (df.index - pd.Timestamp("2009-01-03")).days
    df = df[df['Days'] > 0].dropna()
    
    if "BTC" in ticker:
        # BTC 固定参数
        df['Predicted'] = 10 ** (5.84 * np.log10(df['Days']) - 17.01)
    else:
        # ETH 动态回归
        slope, intercept, _, _, _ = linregress(np.log10(df['Days']), np.log10(df['Close']))
        df['Predicted'] = 10 ** (intercept + slope * np.log10(df['Days']))

    df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
    return df

# --- 4. 绘图逻辑 (核心修改：支持双币种切换) ---
def create_combined_chart(df_btc, df_eth):
    """
    生成包含 BTC 和 ETH 数据的单一图表，使用 updatemenus 切换
    """
    # 颜色定义
    c_price = "#000000"
    c_buy   = "#228b22"
    c_acc   = "#4682b4"
    c_sell  = "#b22222"
    
    # 获取最新数据用于标题字符串
    def get_title_str(df, name):
        if df.empty: return f"{name}: No Data"
        last_p = df['Close'].iloc[-1]
        last_i = df['AHR999'].iloc[-1]
        return f"<b>{name}</b>: ${last_p:,.2f}  |  <b>Deviation Index</b>: {last_i:.4f}"

    title_btc = get_title_str(df_btc, "BTC-USD")
    title_eth = get_title_str(df_eth, "ETH-USD")

    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.65, 0.35], 
        vertical_spacing=0.03
    )
    
    # === 添加 BTC Traces (默认可见: visible=True) ===
    # Trace 0: BTC Price
    fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name="Price", 
                             line=dict(color=c_price, width=1.5), visible=True), row=1, col=1)
    # Trace 1: BTC Model
    fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Predicted'], name="Model", 
                             line=dict(color="purple", width=1, dash='dash'), visible=True), row=1, col=1)
    # Trace 2: BTC Index
    fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['AHR999'], name="Index", 
                             line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # === 添加 ETH Traces (默认隐藏: visible=False) ===
    # Trace 3: ETH Price
    fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Close'], name="Price", 
                             line=dict(color=c_price, width=1.5), visible=False), row=1, col=1)
    # Trace 4: ETH Model
    fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Predicted'], name="Model", 
                             line=dict(color="purple", width=1, dash='dash'), visible=False), row=1, col=1)
    # Trace 5: ETH Index
    fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['AHR999'], name="Index", 
                             line=dict(color="#d35400", width=1.5), visible=False), row=2, col=1)

    # === 关键区域绘制 (背景区域对两者通用) ===
    # 由于 AHR999 是标准化指标，BTC 和 ETH 共用一套阈值区域
    fig.add_hrect(y0=0.0001, y1=0.45, row=2, col=1, fillcolor=c_buy, opacity=0.1, line_width=0, layer="below")
    fig.add_hline(y=0.45, row=2, col=1, line_dash="dot", line_color=c_buy, line_width=1,
                  annotation_text="<b>BUY ZONE (<0.45)</b>", 
                  annotation_position="bottom right", annotation_font=dict(color=c_buy, size=11))
    
    fig.add_hrect(y0=0.45, y1=1.2, row=2, col=1, fillcolor=c_acc, opacity=0.1, line_width=0, layer="below")
    fig.add_hline(y=1.2, row=2, col=1, line_dash="dot", line_color=c_acc, line_width=1,
                  annotation_text="<b>ACCUMULATE (0.45-1.2)</b>", 
                  annotation_position="bottom right", annotation_font=dict(color=c_acc, size=11))
    
    fig.add_hrect(y0=4.0, y1=10000, row=2, col=1, fillcolor=c_sell, opacity=0.1, line_width=0, layer="below")
    fig.add_hline(y=4.0, row=2, col=1, line_dash="dot", line_color=c_sell, line_width=1,
                  annotation_text="<b>HIGH RISK (>4.0)</b>", 
                  annotation_position="top right", annotation_font=dict(color=c_sell, size=11))

    # === 定义 Updatemenus (切换按钮) ===
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                active=0,
                x=0, y=1.08,  # 按钮位置：左上角
                xanchor="left", yanchor="top",
                pad={"r": 10, "t": 10},
                showactive=True,
                bgcolor="white",
                buttons=[
                    # 按钮 1: BTC
                    dict(
                        label="BTC",
                        method="update",
                        args=[
                            # visible: 前3个(BTC)为True，后3个(ETH)为False
                            {"visible": [True, True, True, False, False, False]}, 
                            # layout: 更新标题
                            {"title.text": title_btc}
                        ]
                    ),
                    # 按钮 2: ETH
                    dict(
                        label="ETH",
                        method="update",
                        args=[
                            # visible: 前3个(BTC)为False，后3个(ETH)为True
                            {"visible": [False, False, False, True, True, True]},
                            # layout: 更新标题
                            {"title.text": title_eth}
                        ]
                    )
                ]
            )
        ]
    )

    # 全局设置
    fig.update_yaxes(type="log", row=1, col=1, title="USD (Log)", gridcolor="#f0f0f0")
    fig.update_yaxes(type="log", row=2, col=1, title="Index (Log)", gridcolor="#f0f0f0")
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    
    fig.update_layout(
        title=dict(
            text=title_btc, # 初始标题
            font=dict(size=20, family="Arial"),
            x=0, y=0.98
        ),
        template="plotly_white", 
        height=700, 
        margin=dict(t=100, l=50, r=50, b=50), # 增加顶部边距以容纳按钮
        showlegend=False, 
        xaxis_fixedrange=True, 
        yaxis_fixedrange=True,
        hovermode="x unified"
    )
    return fig

# --- 5. 主程序逻辑 ---

st.title("Market Cycle Monitor")

# 5.1 并行加载数据 (不再依赖 session_state.ticker)
with st.spinner("Fetching market data..."):
    btc_full = get_data("BTC-USD")
    eth_full = get_data("ETH-USD")

# 5.2 确定时间范围（以 BTC 为基准）
if not btc_full.empty:
    min_d, max_d = btc_full.index.min().date(), btc_full.index.max().date()
    def_start = max_d - timedelta(days=365*4)
    if def_start < min_d: def_start = min_d
else:
    min_d = max_d = def_start = datetime.today().date()

# 5.3 布局：仅保留日期滑块
# 将滑块放在 expander 里或者直接展示，这里直接展示保持简洁
dates = st.slider("Analysis Period", min_d, max_d, (def_start, max_d), label_visibility="collapsed")

# 5.4 数据切片
start_ts, end_ts = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])

# 切片函数
def slice_data(df, start, end):
    if df.empty: return df
    return df.loc[(df.index >= start) & (df.index <= end)]

btc_view = slice_data(btc_full, start_ts, end_ts)
eth_view = slice_data(eth_full, start_ts, end_ts)

# 5.5 渲染图表
if not btc_view.empty:
    fig = create_combined_chart(btc_view, eth_view)
    # 重要：移除 config={'staticPlot': True}，否则按钮无法点击
    st.plotly_chart(fig, use_container_width=True) 
else:
    st.error("No data available to display.")
