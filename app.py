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

# --- 3. 核心修复逻辑：预计算与影子坐标轴 ---
def create_chart(df_btc, df_eth):
    # 定义颜色
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    # 确保我们有最新的日期作为基准
    last_date = pd.Timestamp.now()
    if not df_btc.empty: last_date = df_btc.index[-1]
    elif not df_eth.empty: last_date = df_eth.index[-1]

    # --- 辅助函数：计算特定时间窗口下的 Log Y轴范围 ---
    # Log坐标轴的 Range 必须是 [log10(min), log10(max)]
    def get_log_range(df, days_back, col='Close', padding=0.05):
        if df.empty: return [0, 1]
        
        if days_back == 0: # ALL
            target_df = df
        else:
            start_date = last_date - timedelta(days=days_back)
            target_df = df[df.index >= start_date]
        
        if target_df.empty: return [0, 1]
        
        val_min = target_df[col].min()
        val_max = target_df[col].max()
        
        # 防止无效数据
        if val_min <= 0 or pd.isna(val_min): val_min = 1
        if val_max <= 0 or pd.isna(val_max): val_max = 10
        
        # 转换为 Log10 并添加 Padding
        log_min = np.log10(val_min)
        log_max = np.log10(val_max)
        diff = log_max - log_min
        if diff == 0: diff = 0.1
        
        return [log_min - diff * padding, log_max + diff * padding]

    # --- 创建图表结构 ---
    # Rows=2, Cols=1. 默认 Plotly 会创建 y (row1) 和 y2 (row2)
    # 我们将手动添加 y3 (row1, overlay y) 和 y4 (row2, overlay y2) 给 ETH 使用
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.03)

    # --- 绘制 BTC Traces (关联 y 和 y2) ---
    # Visible 初始设为 True
    if not df_btc.empty:
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name="BTC Price", line=dict(color=c_p, width=1.5), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Predicted'], name="BTC Model", line=dict(color="purple", width=1, dash='dash'), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['AHR999'], name="BTC Index", line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # --- 绘制 ETH Traces (关联 y3 和 y4) ---
    # Visible 初始设为 False
    # 注意：yaxis='y3' 是特殊的映射方式
    if not df_eth.empty:
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Close'], name="ETH Price", line=dict(color=c_p, width=1.5), yaxis="y3", visible=False)) # Row 1 Shadow
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Predicted'], name="ETH Model", line=dict(color="purple", width=1, dash='dash'), yaxis="y3", visible=False)) # Row 1 Shadow
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['AHR999'], name="ETH Index", line=dict(color="#d35400", width=1.5), xaxis="x2", yaxis="y4", visible=False)) # Row 2 Shadow

    # --- 绘制背景线 (Row 2) ---
    # 为了简化，背景线画在 y2 上，反正阈值对两个币差不多
    for y_val, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_font=dict(color=c))

    # --- 核心：生成“智能”时间按钮 ---
    # 每个按钮都会同时设置 BTC 的轴范围和 ETH 的轴范围
    time_periods = [
        ("1W", 7), ("2W", 14), ("1M", 30), ("3M", 90), 
        ("6M", 180), ("1Y", 365), ("3Y", 365*3), ("ALL", 0)
    ]
    
    buttons_time = []
    for label, days in time_periods:
        # 1. 计算 X 轴范围
        if days == 0:
            x_range = [None, None] # Autorange for X
        else:
            x_start = last_date - timedelta(days=days)
            x_range = [x_start.strftime("%Y-%m-%d"), last_date.strftime("%Y-%m-%d")]
            
        # 2. 预计算 BTC 的 Y轴范围 (Price & Index)
        btc_p_range = get_log_range(df_btc, days, 'Close')
        btc_i_range = get_log_range(df_btc, days, 'AHR999')
        
        # 3. 预计算 ETH 的 Y轴范围 (Price & Index)
        eth_p_range = get_log_range(df_eth, days, 'Close')
        eth_i_range = get_log_range(df_eth, days, 'AHR999')
        
        # 4. 构建 Args：一次性更新所有轴
        # 注意：yaxis 是 BTC Price, yaxis3 是 ETH Price
        #       yaxis2 是 BTC Index, yaxis4 是 ETH Index
        args_dict = {
            "xaxis.range": x_range,
            "yaxis.range": btc_p_range,  # BTC Price Range
            "yaxis3.range": eth_p_range, # ETH Price Range
            "yaxis2.range": btc_i_range, # BTC Index Range
            "yaxis4.range": eth_i_range, # ETH Index Range
        }
        
        # 如果是 ALL，让 Plotly 自动处理
        if days == 0:
            args_dict = {
                "xaxis.autorange": True,
                "yaxis.autorange": True, "yaxis3.autorange": True,
                "yaxis2.autorange": True, "yaxis4.autorange": True
            }

        buttons_time.append(dict(
            label=label,
            method="relayout",
            args=[args_dict]
        ))

    # --- 生成币种切换按钮 ---
    # 这里需要非常小心地控制 trace 的可见性和 axis 的可见性
    # Traces 顺序: 
    # 0,1,2 -> BTC (Price, Model, Index)
    # 3,4,5 -> ETH (Price, Model, Index)
    
    # 切换到 BTC: 显示 Trace 0-2, 隐藏 3-5; 显示 y/y2, 隐藏 y3/y4
    btn_btc = dict(
        label="BTC", method="update",
        args=[
            {"visible": [True, True, True, False, False, False]}, # Traces
            {
                "title.text": f"<b>BTC-USD</b>: ${df_btc['Close'].iloc[-1]:,.2f} | Index: {df_btc['AHR999'].iloc[-1]:.4f}",
                "yaxis.visible": True, "yaxis2.visible": True,
                "yaxis3.visible": False, "yaxis4.visible": False
            }
        ]
    )
    
    # 切换到 ETH: 显示 Trace 3-5, 隐藏 0-2; 隐藏 y/y2, 显示 y3/y4
    last_eth_price = df_eth['Close'].iloc[-1] if not df_eth.empty else 0
    last_eth_idx = df_eth['AHR999'].iloc[-1] if not df_eth.empty else 0
    btn_eth = dict(
        label="ETH", method="update",
        args=[
            {"visible": [False, False, False, True, True, True]}, # Traces
            {
                "title.text": f"<b>ETH-USD</b>: ${last_eth_price:,.2f} | Index: {last_eth_idx:.4f}",
                "yaxis.visible": False, "yaxis2.visible": False,
                "yaxis3.visible": True, "yaxis4.visible": True
            }
        ]
    )

    # --- 布局设置 ---
    fig.update_layout(
        template="plotly_white",
        height=750,
        margin=dict(t=110, l=40, r=40, b=40),
        title=dict(text=f"<b>BTC-USD</b>: ${df_btc['Close'].iloc[-1]:,.2f} | Index: {df_btc['AHR999'].iloc[-1]:.4f}", x=0, y=0.98),
        hovermode="x unified",
        showlegend=False,
        dragmode="pan", # 允许拖拽
        
        # 按钮容器
        updatemenus=[
            dict(
                type="buttons", direction="left", active=0, x=0, y=1.12,
                buttons=[btn_btc, btn_eth],
                bgcolor="white", bordercolor="#cccccc", borderwidth=1
            ),
            dict(
                type="buttons", direction="left", active=7, x=1, y=1.12, xanchor="right",
                buttons=buttons_time,
                bgcolor="white", bordercolor="#cccccc", borderwidth=1,
                font=dict(size=10)
            )
        ],
        
        # --- 复杂
