import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 全局页面配置 ---
st.set_page_config(
    page_title="Market Cycle Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 样式美化 (CSS) ---
# 隐藏 Streamlit 默认的汉堡菜单和页脚，使界面更像原生 App
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #f8f9fa;
    }
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* 调整滑块样式 */
    .stSlider > div > div > div > div {
        background-color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 稳健的数据获取 (双通道: Yahoo + Coingecko) ---
# 保持之前的稳健逻辑，确保有网就能跑

def fetch_coingecko_data(ticker):
    coin_id = "bitcoin" if "BTC" in ticker else "ethereum"
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': 'max', 'interval': 'daily'}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        if'prices' not in data: return pd.DataFrame()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        return df[['Close']]
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_data(ticker):
    source = "Yahoo Finance"
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty: raise ValueError("Empty")
        # 兼容 yfinance 新旧版本列名差异
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', axis=1, level=0, drop_level=True)
        if'Close' not in df.columns and len(df.columns) >= 1:
             # 假设第一列是 Close
             df = df.iloc[:, 0].to_frame(name='Close')
        else:
            df = df[['Close']]
    except:
        df = fetch_coingecko_data(ticker)
        source = "Coingecko (Backup)"
    
    if df.empty: return df, "Data Error"

    # --- 指标计算逻辑 ---
    df = df.sort_index()
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[df['Close'] > 0]
    
    # 200日定投几何平均成本
    df['Log_Price'] = np.log(df['Close'])
    df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
    
    # 币龄与回归预测
    genesis = pd.Timestamp("2009-01-03")
    df['Days'] = (df.index - genesis).days
    df = df[df['Days'] > 0].dropna()
    
    if "BTC" in ticker:
        # BTC 使用经典的囤币党参数
        slope, intercept = 5.84, -17.01
        df['Predicted'] = 10 ** (slope * np.log10(df['Days']) + intercept)
    else:
        # ETH 使用动态回归
        x = np.log10(df['Days'].values)
        y = np.log10(df['Close'].values)
        slope, intercept, _, _, _ = linregress(x, y)
        df['Predicted'] = 10 ** (intercept + slope * x)

    # AHR999 指数
    df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
    return df, source

# --- 4. 可视化核心 (静态 + 易懂) ---
def create_static_dashboard(df, ticker, start_date, end_date):
    # 1. 数据切片
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_slice = df.loc[mask]
    
    if df_slice.empty:
        st.error("Selected time range has no data.")
        return

    last_price = df_slice['Close'].iloc[-1]
    last_ahr = df_slice['AHR999'].iloc[-1]

    # 2. 配色方案 (专业金融风)
    color_price = "#2c3e50"    # 深蓝灰
    color_pred = "#8e44ad"     # 紫色 (预测线)
    color_buy = "#27ae60"      # 绿色 (抄底)
    color_sell = "#c0392b"     # 红色 (逃顶)
    color_accum = "#2980b9"    # 蓝色 (定投)

    # 3. 创建子图 (上:价格, 下:指标)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.65, 0.35],
        subplot_titles=("Price Action & Valuation Model", "Deviation Index (Market Sentiment)")
    )

    # --- 上半部分：价格 vs 估值 ---
    # 价格线
    fig.add_trace(go.Scatter(x=df_slice.index, y=df_slice['Close'], name="Price",
                             line=dict(color=color_price, width=2)), row=1, col=1)
    # 预测线 (虚线)
    fig.add_trace(go.Scatter(x=df_slice.index, y=df_slice['Predicted'], name="Fair Value",
                             line=dict(color=color_pred, width=2, dash='dash')), row=1, col=1)
    
    # --- 下半部分：指标 (使用背景色带代替线条) ---
    fig.add_trace(go.Scatter(x=df_slice.index, y=df_slice['AHR999'], name="Index",
                             line=dict(color="#d35400", width=2)), row=2, col=1)

    # 关键优化：使用 add_hrect 添加直观的背景色带
    # 抄底区 (<0.45)
    fig.add_hrect(y0=0, y1=0.45, row=2, col=1, 
                  fillcolor=color_buy, opacity=0.15, layer="below", line_width=0,
                  annotation_text="BUY ZONE", annotation_position="top left", annotation_font_color=color_buy)
    # 定投区 (0.45 - 1.2)
    fig.add_hrect(y0=0.45, y1=1.2, row=2, col=1, 
                  fillcolor=color_accum, opacity=0.1, layer="below", line_width=0,
                  annotation_text="ACCUMULATE", annotation_position="top left", annotation_font_color=color_accum)
    # 泡沫区 (>4.0) - 只有当数据真的触及时才显示，避免压缩视图
    if df_slice['AHR999'].max() > 3.0:
        fig.add_hrect(y0=4.0, y1=100, row=2, col=1, 
                      fillcolor=color_sell, opacity=0.15, layer="below", line_width=0,
                      annotation_text="SELL ZONE", annotation_position="bottom left", annotation_font_color=color_sell)

    # --- 布局优化 ---
    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=50, l=50, r=50, b=50),
        showlegend=False, # 隐藏图例，让图表更纯粹，依靠标题和颜色识别
        title=dict(
            text=f"<b>{ticker}</b>: ${last_price:,.0f} | <b>Index</b>: {last_ahr:.2f}",
            x=0.05, y=0.98, xanchor='left',
            font=dict(size=20, family="Arial")
        )
    )

    # 坐标轴设置
    fig.update_yaxes(type="log", title="Price (USD)", row=1, col=1, gridcolor="#eee")
    fig.update_yaxes(title="Deviation", row=2, col=1, gridcolor="#eee", zeroline=False)
    fig.update_xaxes(showgrid=True, gridcolor="#eee")

    # 输出静态图表
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True}) 

# --- 5. 主控制面板 (Sidebar + Layout) ---

# 侧边栏：全局控制
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. 资产选择 (Radio比Tab更适合作为全局开关)
    ticker_option = st.radio("Select Asset", ["BTC-USD", "ETH-USD"], index=0)
    
    st.divider()
    
    # 2. 加载数据
    with st.spinner("Fetching Data..."):
        df_full, source_note = get_data(ticker_option)
    
    # 3. 时间切片器 (独立于图表)
    st.subheader("📅 Time Range")
    
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()
    default_start = max_date - timedelta(days=365*4) # 默认看最近4年
    
    if default_start < min_date: default_start = min_date

    # 滑块控件
    date_range = st.slider(
        "Zoom Level",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD"
    )

    st.divider()
    st.caption(f"Data Source: {source_note}")
    st.caption("Mode: Static View (Non-interactive)")

# --- 6. 主体显示 ---

# 使用容器包裹，增加一点白色背景卡片感
with st.container():
    if not df_full.empty:
        # 调用绘图函数
        create_static_dashboard(df_full, ticker_option, date_range[0], date_range[1])
        
        # 底部状态解释
        st.markdown(f"""
        ---
        **How to read this chart:**
        - **Top Panel**: The <span style='color:#2c3e50'><b>Dark Line</b></span> is the actual price. The <span style='color:#8e44ad'><b>Purple Dashed Line</b></span> is the "Fair Value" model.
        - **Bottom Panel**: The Deviation Index.
            - <span style='color:#27ae60; background-color:#eafaf1; padding:2px 5px; border-radius:3px;'><b>Green Zone (<0.45)</b></span>: Historically the best time to buy.
            - <span style='color:#2980b9; background-color:#ebf5fb; padding:2px 5px; border-radius:3px;'><b>Blue Zone (0.45-1.2)</b></span>: Good for Dollar Cost Averaging (DCA).
            - <span style='color:#c0392b; background-color:#fdedec; padding:2px 5px; border-radius:3px;'><b>Red Zone (>4.0)</b></span>: Historically overheated (Sell signal).
        """, unsafe_allow_html=True)
    else:
        st.error("Unable to load data. Please check your internet connection.")

