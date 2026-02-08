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
    /* 隐藏 Plotly 自带的 Modebar，让界面更像 Coinglass */
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. 数据获取 (保持不变) ---
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

# --- 3. 绘图逻辑 (Coinglass 风格) ---
def create_chart(df_btc, df_eth):
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    # 建立双轴框架 (Row 1: Price, Row 2: Index)
    # shared_xaxes=True 会把 Range Slider 放在最底部
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

    # --- 1. 绘制 BTC Traces (默认可见) ---
    if not df_btc.empty:
        # Price (Row 1, y)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name="BTC Price", 
                                 line=dict(color=c_p, width=1.5), visible=True), row=1, col=1)
        # Model (Row 1, y)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Predicted'], name="BTC Model", 
                                 line=dict(color="purple", width=1, dash='dash'), visible=True), row=1, col=1)
        # Index (Row 2, y2)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['AHR999'], name="BTC Index", 
                                 line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # --- 2. 绘制 ETH Traces (默认隐藏，使用影子轴 y3, y4) ---
    if not df_eth.empty:
        # Price (Row 1, y3 - Shadow)
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Close'], name="ETH Price", 
                                 line=dict(color=c_p, width=1.5), yaxis="y3", visible=False), row=1, col=1) # 注意这里不需要 col/row 参数因为 yaxis="y3" 已经指定了映射
        # Model (Row 1, y3 - Shadow)
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Predicted'], name="ETH Model", 
                                 line=dict(color="purple", width=1, dash='dash'), yaxis="y3", visible=False))
        # Index (Row 2, y4 - Shadow)
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['AHR999'], name="ETH Index", 
                                 line=dict(color="#d35400", width=1.5), xaxis="x2", yaxis="y4", visible=False)) # 需要指定 xaxis="x2" 对应 Row 2

    # --- 3. 绘制指标线 (Row 2) ---
    for y_val, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_position="top left", annotation_font=dict(color=c, size=10))

    # --- 4. 按钮定义 (只保留币种切换) ---
    btn_btc = dict(
        label="Bitcoin (BTC)", method="update",
        args=[
            {"visible": [True, True, True, False, False, False]}, # Traces visibility
            {
                "title.text": f"<b>BTC-USD</b> Market Cycle",
                "yaxis.visible": True, "yaxis2.visible": True, # BTC Axes
                "yaxis3.visible": False, "yaxis4.visible": False # ETH Axes
            }
        ]
    )
    
    btn_eth = dict(
        label="Ethereum (ETH)", method="update",
        args=[
            {"visible": [False, False, False, True, True, True]}, 
            {
                "title.text": f"<b>ETH-USD</b> Market Cycle",
                "yaxis.visible": False, "yaxis2.visible": False, # BTC Axes
                "yaxis3.visible": True, "yaxis4.visible": True # ETH Axes
            }
        ]
    )

    # --- 5. 布局核心设置 ---
    fig.update_layout(
        template="plotly_white",
        height=700, # 稍微高一点以容纳 Slider
        margin=dict(t=80, l=40, r=40, b=40), # 顶部留出标题空间
        title=dict(text=f"<b>BTC-USD</b> Market Cycle", x=0.01, y=0.96, font=dict(size=20)),
        hovermode="x unified",
        showlegend=False,
        dragmode="pan", # 默认拖拽模式为平移
        
        # 顶部按钮组
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                active=0,
                x=0.01, y=1.08, # 放在标题上方
                buttons=[btn_btc, btn_eth],
                bgcolor="white", bordercolor="#e0e0e0", borderwidth=1,
                font=dict(size=12)
            )
        ],
        
        # --- 关键：X轴与 Range Slider 设置 ---
        xaxis=dict(
            anchor="y2", # 锚定到底部图表
            type="date",
            rangeslider=dict(
                visible=True, # 开启滑块
                thickness=0.08, # 滑块高度占比
                bgcolor="#f8f9fa", # 滑块背景色
                bordercolor="#dee2e6",
                borderwidth=1
            ),
            rangeselector=dict(visible=False) # 彻底禁用原生的 rangeselector 按钮
        ),
        
        # --- Y轴设置 (BTC) ---
        yaxis=dict(
            domain=[0.35, 1], # 上半部分 65%
            type="log", title="Price (USD)", 
            fixedrange=False, # 允许纵向缩放 (虽然 Rangeslider 不会自动触发)
            showgrid=True, gridcolor="#f0f0f0"
        ),
        yaxis2=dict(
            domain=[0, 0.30], # 下半部分 30%
            type="log", title="Index",
            showgrid=True, gridcolor="#f0f0f0"
        ),

        # --- Y轴设置 (ETH - 影子轴) ---
        # 必须与 BTC 的 domain 完全一致，且 overlaying 对应轴
        yaxis3=dict(
            domain=[0.35, 1], anchor="x", overlaying="y", side="left",
            type="log", title="Price (USD)", visible=False, showgrid=False
        ),
        yaxis4=dict(
            domain=[0, 0.30], anchor="x", overlaying="y2", side="left",
            type="log", title="Index", visible=False, showgrid=False
        )
    )

    return fig

# --- 4. 主程序 ---
st.title("Market Cycle Monitor")
with st.spinner("Loading market data..."):
    btc_df, eth_df = get_data("BTC-USD"), get_data("ETH-USD")

if not btc_df.empty:
    fig = create_chart(btc_df, eth_df)
    # config 中移除多余按钮，保持界面清爽
    st.plotly_chart(fig, use_container_width=True, 
                    config={
                        'displayModeBar': False, # 隐藏顶部工具栏
                        'scrollZoom': True,      # 允许滚轮缩放
                        'showAxisRangeEntryBoxes': False
                    })
else:
    st.error("Data loading failed.")
