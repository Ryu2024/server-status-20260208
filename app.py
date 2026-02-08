import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 页面基础配置 (无 Emoji) ---
st.set_page_config(
    page_title="Market Cycle Monitor",
    layout="wide"
)

# --- 2. CSS 样式微调 ---
st.markdown("""
<style>
    /* 隐藏顶部红线和汉堡菜单，打造原生应用感 */
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* 调整滑块颜色为深灰，更显专业 */
    .stSlider > div > div > div > div {
        background-color: #4a4a4a;
    }
    /* 调整指标卡片样式 */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 数据获取模块 (双通道) ---
def fetch_coingecko_data(ticker):
    """备用数据源"""
    coin_id = "bitcoin" if "BTC" in ticker else "ethereum"
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': 'max', 'interval': 'daily'}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        if'prices' not in data: return pd.DataFrame()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        return df[['Close']]
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_data(ticker):
    """主数据源逻辑"""
    source = "Yahoo Finance"
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty: raise ValueError("Empty")
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', axis=1, level=0, drop_level=True)
        if'Close' not in df.columns:
             df = df.iloc[:, 0].to_frame(name='Close')
    except:
        df = fetch_coingecko_data(ticker)
        source = "Coingecko"
    
    if df.empty: return df, "Data Unavailable"

    # --- 数据清洗与计算 ---
    df = df.sort_index()
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[df['Close'] > 0]
    
    # 200日几何平均
    df['Log_Price'] = np.log(df['Close'])
    df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
    
    # 回归模型
    genesis = pd.Timestamp("2009-01-03")
    df['Days'] = (df.index - genesis).days
    df = df[df['Days'] > 0].dropna()
    
    if "BTC" in ticker:
        slope, intercept = 5.84, -17.01
        df['Predicted'] = 10 ** (slope * np.log10(df['Days']) + intercept)
    else:
        x = np.log10(df['Days'].values)
        y = np.log10(df['Close'].values)
        slope, intercept, _, _, _ = linregress(x, y)
        df['Predicted'] = 10 ** (intercept + slope * x)

    # AHR999 指数
    df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
    return df, source

def create_pure_chart(df):
    # 颜色定义
    c_price = "#000000"
    c_buy   = "#228b22"  # 绿色
    c_acc   = "#4682b4"  # 蓝色
    c_sell  = "#b22222"  # 红色
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.65, 0.35], 
        vertical_spacing=0.03
    )
    
    # --- 上图：价格 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", 
                             line=dict(color=c_price, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name="Model", 
                             line=dict(color="purple", width=1, dash='dash')), row=1, col=1)
    
    # --- 下图：指标 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['AHR999'], name="Index", 
                             line=dict(color="#d35400", width=1.5)), row=2, col=1)
    
    # ------------------------------------------------------------------
    # 核心修复区：精确控制文字位置，使其落入对应色块中
    # ------------------------------------------------------------------

    # 1. 抄底区 (绿色, < 0.45)
    # 背景色
    fig.add_hrect(y0=0.001, y1=0.45, row=2, col=1, 
                  fillcolor=c_buy, opacity=0.1, line_width=0, layer="below")
    # 分界线 & 文字
    fig.add_hline(y=0.45, row=2, col=1, 
                  line_dash="dot", line_color=c_buy, line_width=1,
                  annotation_text="<b>BUY ZONE (<0.45)</b>", 
                  # 关键：Bottom Right -> 让文字位于线的下方（绿色区域内）
                  annotation_position="bottom right",
                  annotation_font=dict(color=c_buy, size=10))
    
    # 2. 定投区 (蓝色, 0.45 - 1.2)
    # 背景色
    fig.add_hrect(y0=0.45, y1=1.2, row=2, col=1, 
                  fillcolor=c_acc, opacity=0.1, line_width=0, layer="below")
    # 分界线 (上限 1.2) & 文字
    fig.add_hline(y=1.2, row=2, col=1, 
                  line_dash="dot", line_color=c_acc, line_width=1,
                  annotation_text="<b>ACCUMULATE (0.45-1.2)</b>", 
                  # 关键：Bottom Right -> 让文字位于1.2线的下方（蓝色区域内）
                  annotation_position="bottom right",
                  annotation_font=dict(color=c_acc, size=10))
    
    # 3. 顶部风险区 (红色, > 4.0)
    # 背景色
    fig.add_hrect(y0=4.0, y1=1000, row=2, col=1, 
                  fillcolor=c_sell, opacity=0.1, line_width=0, layer="below")
    # 分界线 & 文字
    fig.add_hline(y=4.0, row=2, col=1, 
                  line_dash="dot", line_color=c_sell, line_width=1,
                  annotation_text="<b>HIGH RISK (>4.0)</b>", 
                  # 关键：Top Right -> 让文字位于4.0线的上方（红色区域内）
                  annotation_position="top right",
                  annotation_font=dict(color=c_sell, size=10))

    # ------------------------------------------------------------------

    # 全局 Log 坐标设置
    fig.update_yaxes(type="log", row=1, col=1, title="Price (USD)", gridcolor="#f0f0f0")
    fig.update_yaxes(type="log", row=2, col=1, title="Index Value", gridcolor="#f0f0f0")
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    
    # 布局微调
    fig.update_layout(
        template="plotly_white", 
        height=700, 
        margin=dict(t=10, l=50, r=50, b=50), 
        showlegend=False, 
        xaxis_fixedrange=True, 
        yaxis_fixedrange=True
    )
    return fig

# --- 5. 主界面逻辑 (Top Control Layout) ---

st.title("Statistical Deviation Monitor")

# 初始化数据 (为了获取时间范围)
# 这里先加载一次数据，用于确定滑块的范围，但还不绘图
with st.spinner("Initializing..."):
    # 默认先拿 BTC 获取时间范围
    temp_df, _ = get_data("BTC-USD") 
    if not temp_df.empty:
        min_date = temp_df.index.min().date()
        max_date = temp_df.index.max().date()
        default_start = max_date - timedelta(days=365*3) # 默认3年
    else:
        min_date, max_date = datetime.today().date(), datetime.today().date()
        default_start = min_date

# --- 顶部控制器区域 (Columns) ---
# 比例 1:3，左边选币，右边选时间
c_ctrl_1, c_ctrl_2 = st.columns([1, 3])

with c_ctrl_1:
    ticker_opt = st.selectbox("Asset", ["BTC-USD", "ETH-USD"], index=0)

with c_ctrl_2:
    # 必须确保 default_start 有效
    if default_start < min_date: default_start = min_date
    
    date_range = st.slider(
        "Analysis Period",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD"
    )

# --- 数据处理与渲染 ---
df_full, source_note = get_data(ticker_opt)

if not df_full.empty:
    # 1. 根据滑块切片数据
    mask = (df_full.index >= pd.to_datetime(date_range[0])) & (df_full.index <= pd.to_datetime(date_range[1]))
    df_display = df_full.loc[mask]

    if not df_display.empty:
        # 2. 生成图表
        fig, price, ahr = create_chart(df_display, ticker_opt)

        # 3. 显示顶部简报 (Metric)
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${price:,.2f}")
        m2.metric("Deviation Index", f"{ahr:.4f}")
        m3.caption(f"Source: {source_note} | Mode: Log-Log Regression")

        # 4. 渲染图表 (静态)
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
    else:
        st.warning("No data in selected range.")
else:
    st.error("Connection failed. Please check network.")
