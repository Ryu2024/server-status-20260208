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
    /* 强制隐藏 Plotly 顶部的 Modebar (悬浮工具栏) */
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

# --- 3. 绘图逻辑 (修复滑块与按钮) ---
def create_chart(df_btc, df_eth):
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    # 1. 创建子图
    # vertical_spacing 稍微调大一点，避免滑块遮挡
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)

    # 2. 绘制 BTC (Row 1 & Row 2)
    if not df_btc.empty:
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name="BTC Price", line=dict(color=c_p, width=1.5), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Predicted'], name="BTC Model", line=dict(color="purple", width=1, dash='dash'), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['AHR999'], name="BTC Index", line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # 3. 绘制 ETH (影子轴, 默认隐藏)
    if not df_eth.empty:
        # 注意：这里不需要指定 xaxis，因为 shared_xaxes 会自动处理 x 轴同步
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Close'], name="ETH Price", line=dict(color=c_p, width=1.5), yaxis="y3", visible=False))
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Predicted'], name="ETH Model", line=dict(color="purple", width=1, dash='dash'), yaxis="y3", visible=False))
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['AHR999'], name="ETH Index", line=dict(color="#d35400", width=1.5), yaxis="y4", visible=False))

    # 4. 指标线
    for y_val, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_position="top left", annotation_font=dict(color=c, size=10))

    # 5. 定义按钮 (仅 BTC/ETH 切换)
    btn_btc = dict(
        label="BTC", method="update",
        args=[{"visible": [True, True, True, False, False, False]}, 
              {"title.text": f"<b>BTC-USD</b>", "yaxis.visible": True, "yaxis2.visible": True, "yaxis3.visible": False, "yaxis4.visible": False}]
    )
    btn_eth = dict(
        label="ETH", method="update",
        args=[{"visible": [False, False, False, True, True, True]}, 
              {"title.text": f"<b>ETH-USD</b>", "yaxis.visible": False, "yaxis2.visible": False, "yaxis3.visible": True, "yaxis4.visible": True}]
    )

    # --- 关键修改开始 ---

    # 6. 先对所有 X 轴执行“清零”操作：关闭所有 rangeselector (1W/1M按钮) 和 rangeslider
    fig.update_xaxes(
        rangeselector=dict(visible=False), # 彻底杀死 1W, 1M 按钮
        rangeslider=dict(visible=False),   # 默认先关掉滑块
        fixedrange=False                   # 确保允许拖动
    )

    # 7. 仅对最底部的 X 轴 (Row 2) 开启滑块
    # 在 make_subplots 中，底部的 axis 实际上是 anchor 在 y2 上的那个
    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.05,  # 滑块厚度
            bgcolor="#f4f4f4"
        ),
        row=2, col=1 # 精确打击底部子图
    )

    # --- 关键修改结束 ---

    # 8. 布局设置
    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=80, l=40, r=40, b=60), # 底部留白给滑块
        title=dict(text="<b>BTC-USD</b>", x=0.01, y=0.96),
        hovermode="x unified",
        showlegend=False,
        dragmode="pan", # 默认平移
        
        updatemenus=[dict(
            type="buttons", direction="left", active=0, x=0.01, y=1.08,
            buttons=[btn_btc, btn_eth], bgcolor="white", bordercolor="#e0e0e0", borderwidth=1
        )],
        
        # Y轴配置
        yaxis=dict(domain=[0.35, 1], type="log", title="Price"),
        yaxis2=dict(domain=[0, 0.30], type="log", title="Index"),
        
        # 影子轴配置 (保持不变)
        yaxis3=dict(domain=[0.35, 1], anchor="x", overlaying="y", side="left", type="log", visible=False, showgrid=False),
        yaxis4=dict(domain=[0, 0.30], anchor="x", overlaying="y2", side="left", type="log", visible=False, showgrid=False)
    )

    return fig

# --- 4. 主程序 ---
st.title("Market Cycle Monitor")
with st.spinner("Loading..."):
    btc_df, eth_df = get_data("BTC-USD"), get_data("ETH-USD")

if not btc_df.empty:
    fig = create_chart(btc_df, eth_df)
    
    st.plotly_chart(fig, use_container_width=True, 
                    config={
                        'displayModeBar': False,  # 隐藏右上角 Plotly 工具栏
                        'scrollZoom': True,       # 允许滚轮
                        'showAxisRangeEntryBoxes': False
                    })
else:
    st.error("No Data")
