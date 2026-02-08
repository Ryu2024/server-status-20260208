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

# 初始化 Session State (用于记录当前选中的币种)
if'ticker' not in st.session_state:
    st.session_state.ticker = "BTC-USD"

# --- 2. 深度样式定制 (CSS) ---
st.markdown("""
<style>
    /* 隐藏默认头部和菜单 */
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 按钮基础样式 - 去除圆角，增加极简感 */
    div.stButton > button {
        border-radius: 4px;
        border: 1px solid #ccc;
        font-weight: bold;
        height: 40px; 
        width: 100%;
    }
    
    /* 滑块样式微调 */
    div[data-testid="stSlider"] {
        padding-top: 0px;
    }
</style>
""", unsafe_allow_html=True)

# 动态 CSS：根据当前选中状态，高亮对应按钮
# 注意：这里使用 nth-of-type 选择器来定位 Columns 中的按钮
if st.session_state.ticker == "BTC-USD":
    st.markdown("""<style>
    /* 第1个 Column (BTC) 的按钮变黑 */
    div[data-testid="column"]:nth-of-type(1) div.stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-color: #000000 !important;
    }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
    /* 第2个 Column (ETH) 的按钮变黑 */
    div[data-testid="column"]:nth-of-type(2) div.stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-color: #000000 !important;
    }
    </style>""", unsafe_allow_html=True)

# --- 3. 数据逻辑 (双通道: Yahoo + Coingecko) ---
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
    """主数据获取与计算逻辑"""
    try:
        # 尝试 Yahoo
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty: raise ValueError
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('Close', axis=1, level=0, drop_level=True)
        if'Close' not in df.columns: df = df.iloc[:, 0].to_frame('Close')
    except: 
        # 失败切 Coingecko
        df = fetch_coingecko(ticker)
    
    if df.empty: return df, "Error"

    # 清洗与计算
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
    return df, "OK"

# --- 4. 绘图逻辑 (修复对齐与返回值) ---
def create_final_chart(df, ticker_name):
    """生成最终图表"""
    # 颜色定义
    c_price = "#000000"
    c_buy   = "#228b22"
    c_acc   = "#4682b4"
    c_sell  = "#b22222"
    
    # 获取最新数据用于标题
    last_p = df['Close'].iloc[-1]
    last_i = df['AHR999'].iloc[-1]
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.65, 0.35], 
        vertical_spacing=0.03
    )
    
    # --- 上图：价格 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color=c_price, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name="Model", line=dict(color="purple", width=1, dash='dash')), row=1, col=1)
    
    # --- 下图：指标 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['AHR999'], name="Index", line=dict(color="#d35400", width=1.5)), row=2, col=1)
    
    # --- 关键区域绘制 (带修正后的文字对齐) ---
    
    # 1. Buy Zone (<0.45) -> 文字在右下
    fig.add_hrect(y0=0.0001, y1=0.45, row=2, col=1, fillcolor=c_buy, opacity=0.1, line_width=0, layer="below")
    fig.add_hline(y=0.45, row=2, col=1, line_dash="dot", line_color=c_buy, line_width=1,
                  annotation_text="<b>BUY ZONE (<0.45)</b>", 
                  annotation_position="bottom right", annotation_font=dict(color=c_buy, size=11))
    
    # 2. Accum Zone (0.45 - 1.2) -> 文字在右下
    fig.add_hrect(y0=0.45, y1=1.2, row=2, col=1, fillcolor=c_acc, opacity=0.1, line_width=0, layer="below")
    fig.add_hline(y=1.2, row=2, col=1, line_dash="dot", line_color=c_acc, line_width=1,
                  annotation_text="<b>ACCUMULATE (0.45-1.2)</b>", 
                  annotation_position="bottom right", annotation_font=dict(color=c_acc, size=11))
    
    # 3. High Zone (>4.0) -> 文字在右上 (防止出界)
    fig.add_hrect(y0=4.0, y1=10000, row=2, col=1, fillcolor=c_sell, opacity=0.1, line_width=0, layer="below")
    fig.add_hline(y=4.0, row=2, col=1, line_dash="dot", line_color=c_sell, line_width=1,
                  annotation_text="<b>HIGH RISK (>4.0)</b>", 
                  annotation_position="top right", annotation_font=dict(color=c_sell, size=11))

    # 全局设置 (双对数坐标)
    fig.update_yaxes(type="log", row=1, col=1, title="USD (Log)", gridcolor="#f0f0f0")
    fig.update_yaxes(type="log", row=2, col=1, title="Index (Log)", gridcolor="#f0f0f0")
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    
    # 布局美化
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker_name}</b>: ${last_p:,.2f}  |  <b>Deviation Index</b>: {last_i:.4f}",
            font=dict(size=20, family="Arial"),
            x=0,
            y=0.98
        ),
        template="plotly_white", 
        height=750, 
        margin=dict(t=60, l=50, r=50, b=50), 
        showlegend=False, 
        xaxis_fixedrange=True, 
        yaxis_fixedrange=True,
        hovermode="x unified"
    )
    return fig

# --- 5. 主程序逻辑 ---

st.title("Market Cycle Monitor")

# 5.1 数据预加载 (为了滑块范围)
init_df, _ = get_data(st.session_state.ticker)

if not init_df.empty:
    min_d, max_d = init_df.index.min().date(), init_df.index.max().date()
    # 默认显示最近 4 年
    def_start = max_d - timedelta(days=365*4)
    if def_start < min_d: def_start = min_d
else:
    # 兜底日期
    min_d = max_d = def_start = datetime.today().date()

# 5.2 顶部控制栏布局
# [BTC按钮] [ETH按钮] [空白] [时间滑块]
c_btc, c_eth, c_space, c_slider = st.columns([0.8, 0.8, 0.2, 4])

with c_btc:
    if st.button("BTC", use_container_width=True):
        st.session_state.ticker = "BTC-USD"
        st.rerun()

with c_eth:
    if st.button("ETH", use_container_width=True):
        st.session_state.ticker = "ETH-USD"
        st.rerun()

with c_slider:
    dates = st.slider("Analysis Period", min_d, max_d, (def_start, max_d), label_visibility="collapsed")

# 5.3 数据切片与渲染
df_full, status = get_data(st.session_state.ticker)

if not df_full.empty:
    # 切片
    mask = (df_full.index >= pd.to_datetime(dates[0])) & (df_full.index <= pd.to_datetime(dates[1]))
    df_view = df_full.loc[mask]
    
    if not df_view.empty:
        # 生成图表 (只返回 fig)
        fig = create_final_chart(df_view, st.session_state.ticker)
        # 渲染 (静态模式)
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
    else:
        st.warning("No data in the selected time range.")
else:
    st.error("Unable to connect to market data sources (Yahoo/Coingecko).")
