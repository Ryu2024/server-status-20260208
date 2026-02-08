import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta

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
    # 尝试 Yahoo
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

    # 尝试 Coingecko
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
    
    # 清洗数据
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

# --- 3. 绘图逻辑 (核心修改) ---
def create_chart(df_btc, df_eth):
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    # 辅助函数：获取标题
    def get_t(df, n):
        if df.empty: return f"{n}: No Data"
        return f"<b>{n}</b>: ${df['Close'].iloc[-1]:,.2f} | Index: {df['AHR999'].iloc[-1]:.4f}"

    # 1. 计算时间范围 (基于数据真实存在的最后一天)
    # 我们以 BTC 的时间为基准，如果 BTC 没数据则用 ETH，都为空则用今天
    ref_df = df_btc if not df_btc.empty else (df_eth if not df_eth.empty else pd.DataFrame())
    
    if not ref_df.empty:
        last_date = ref_df.index[-1]
    else:
        # 如果完全没数据，无需渲染后续
        return go.Figure()

    # 预计算各个时间窗口的开始日期 (转为字符串给 Plotly 用)
    d_max = last_date
    d_1w = d_max - timedelta(days=7)
    d_2w = d_max - timedelta(days=14)
    d_1m = d_max - timedelta(days=30)
    d_3m = d_max - timedelta(days=90)
    d_6m = d_max - timedelta(days=180)
    d_1y = d_max - timedelta(days=365)
    d_3y = d_max - timedelta(days=365*3)

    # 2. 创建图表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.03)

    # 3. 绘制 Traces
    for df, vis, name in [(df_btc, True, "BTC"), (df_eth, False, "ETH")]:
        d = df if not df.empty else pd.DataFrame({'Close':[],'Predicted':[],'AHR999':[]})
        fig.add_trace(go.Scatter(x=d.index, y=d['Close'], name="Price", line=dict(color=c_p, width=1.5), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['Predicted'], name="Model", line=dict(color="purple", width=1, dash='dash'), visible=vis), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d['AHR999'], name="Index", line=dict(color="#d35400", width=1.5), visible=vis), row=2, col=1)

    # 4. 绘制背景区域
    for y, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_font=dict(color=c))

    # 5. 定义 Updatemenus (按钮组)
    # 左侧：切换币种
    # 右侧：切换时间 (手动计算 range，不依赖 rangeselector)
    
    # 币种切换逻辑
    button_btc = dict(label="BTC", method="update", args=[{"visible": [True, True, True, False, False, False]}, {"title.text": get_t(df_btc, "BTC-USD")}])
    button_eth = dict(label="ETH", method="update", args=[{"visible": [False, False, False, True, True, True]}, {"title.text": get_t(df_eth, "ETH-USD")}])

    # 时间切换逻辑 (核心修复)
    # 强制设置 xaxis.range 和 yaxis.autorange
    def time_btn(label, start_date):
        return dict(
            label=label,
            method="relayout",
            args=[{
                "xaxis.range": [start_date, d_max], 
                "xaxis.autorange": False,
                "yaxis.autorange": True,  # 强制 Y 轴重置
                "yaxis2.autorange": True
            }]
        )
    
    time_buttons = [
        time_btn("1W", d_1w),
        time_btn("2W", d_2w),
        time_btn("1M", d_1m),
        time_btn("3M", d_3m),
        time_btn("6M", d_6m),
        time_btn("1Y", d_1y),
        time_btn("3Y", d_3y),
        dict(label="ALL", method="relayout", args=[{"xaxis.autorange": True, "yaxis.autorange": True, "yaxis2.autorange": True}])
    ]

    fig.update_layout(
        # 禁用交互
        dragmode=False,
        
        updatemenus=[
            # 左侧按钮组：BTC / ETH
            dict(
                type="buttons", direction="left", active=0, x=0, y=1.12,
                buttons=[button_btc, button_eth],
                bgcolor="white", bordercolor="#cccccc", borderwidth=1
            ),
            # 右侧按钮组：时间选择 (替代了 rangeselector)
            dict(
                type="buttons", direction="left", active=7, x=1, y=1.12, xanchor="right",
                buttons=time_buttons,
                bgcolor="white", bordercolor="#cccccc", borderwidth=1,
                font=dict(size=10) # 缩小字体防止拥挤
            )
        ],
        
        hovermode="x unified",
        template="plotly_white",
        height=750,
        margin=dict(t=110, l=40, r=40, b=40),
        title=dict(text=get_t(df_btc, "BTC-USD"), x=0, y=0.98),
        showlegend=False,
    )

    # 6. 坐标轴设置
    # X轴：锁死交互
    fig.update_xaxes(fixedrange=True, row=1, col=1) 
    fig.update_xaxes(fixedrange=True, row=2, col=1)

    # Y轴：解锁 fixedrange (为了 autorange 生效)，但因为 dragmode=False，用户依然无法手动拖拽
    fig.update_yaxes(type="log", title="USD", row=1, col=1, fixedrange=False)
    fig.update_yaxes(type="log", title="Index", row=2, col=1, fixedrange=False)
    
    return fig

# --- 4. 主程序 ---
st.title("Market Cycle Monitor")
with st.spinner("Syncing data..."):
    btc_df, eth_df = get_data("BTC-USD"), get_data("ETH-USD")

if not btc_df.empty:
    fig = create_chart(btc_df, eth_df)
    # 彻底禁用 Streamlit 端的交互配置
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False, 'staticPlot': False})
else:
    st.error("Unable to load data.")
