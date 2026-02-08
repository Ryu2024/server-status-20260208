import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests  # 新增：用于请求备用接口
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- 1. 页面配置 ---
st.set_page_config(page_title="Crypto Dashboard Pro", layout="wide")

# --- 2. 增强型数据获取模块 (双通道) ---

def fetch_coingecko_data(ticker):
    """
    备用通道：从 Coingecko 获取历史数据 (无需 Key，真实数据)
    """
    # 映射 Ticker 到 Coingecko ID
    coin_id = "bitcoin" if "BTC" in ticker else "ethereum"
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    
    # 请求参数：对美元，最大时间跨度，按天
    params = {
        'vs_currency': 'usd',
        'days': 'max',
        'interval': 'daily'
    }
    
    try:
        # 增加 User-Agent 伪装，防止被识别为爬虫
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if'prices' not in data:
            return pd.DataFrame()
            
        # Coingecko 返回的是 [[timestamp, price], ...]
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        
        # 简单清洗
        df = df[~df.index.duplicated(keep='last')]
        return df
        
    except Exception as e:
        print(f"Coingecko fallback failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600) # 缓存10分钟
def get_robust_data(ticker):
    """
    智能路由数据获取：优先 Yahoo，失败则自动切换 Coingecko
    """
    df = pd.DataFrame()
    source_used = "Yahoo"
    
    # --- 通道 1: Yahoo Finance ---
    try:
        # 尝试使用 Ticker 对象，有时比 download 更稳定
        dat = yf.Ticker(ticker)
        df = dat.history(period="max")
        
        if df.empty:
            raise ValueError("Yahoo returned empty data")
            
    except Exception as e_yahoo:
        # --- 通道 2: Coingecko (Failover) ---
        print(f"Yahoo failed ({e_yahoo}), switching to Coingecko...")
        try:
            df = fetch_coingecko_data(ticker)
            source_used = "Coingecko"
        except Exception as e_cg:
            print(f"All sources failed: {e_cg}")
    
    return df, source_used

def calculate_metrics(ticker):
    """数据计算与清洗逻辑"""
    raw_df, source = get_robust_data(ticker)
    
    if raw_df.empty:
        return raw_df, f"Error: Data Unavailable ({ticker})", 0, 0, "N/A"

    # 标准化列名 (兼容不同来源)
    if'Close' not in raw_df.columns:
        # 模糊匹配
        cols = [c for c in raw_df.columns if'Close' in str(c)]
        if cols:
            raw_df = raw_df[[cols[0]]].rename(columns={cols[0]: 'Close'})
        else:
            return pd.DataFrame(), "Column Error", 0, 0, "Error"

    df = raw_df[['Close']].copy().sort_index()
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[df['Close'] > 0].dropna()
    
    # 核心指标计算
    df['Log_Price'] = np.log(df['Close'])
    df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
    
    genesis_date = pd.Timestamp("2009-01-03")
    df['Days'] = (df.index - genesis_date).days
    df = df[df['Days'] > 0]

    # 预测模型
    if "BTC" in ticker:
        slope = 5.84
        intercept = -17.01
        log_days = np.log10(df['Days'])
        df['Predicted'] = 10 ** (slope * log_days + intercept)
        note = f"Source: {source} | Model: Power Law"
    else:
        # ETH 动态回归
        valid_data = df.dropna()
        if len(valid_data) > 10:
            x = np.log10(valid_data['Days'].values)
            y = np.log10(valid_data['Close'].values)
            slope, intercept, _, _, _ = linregress(x, y)
            df['Predicted'] = 10 ** (intercept + slope * np.log10(df['Days']))
            note = f"Source: {source} | Model: Reg (Beta {slope:.2f})"
        else:
            df['Predicted'] = np.nan
            note = f"Source: {source} | Data Insufficient"

    # AHR999 / 偏离度
    df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
    
    # 最新状态
    last_row = df.iloc[-1]
    current_price = last_row['Close']
    current_ahr = last_row['AHR999'] if not np.isnan(last_row['AHR999']) else 0
    
    return df, note, current_price, current_ahr

@st.cache_data(ttl=3600)
def load_all_market_data():
    """一次性加载所有数据"""
    data_map = {}
    tickers = ["BTC-USD", "ETH-USD"]
    
    for t in tickers:
        df, note, price, ahr = calculate_metrics(t)
        
        state_text = "ZONE H (High)"
        color_hex = "#dc3545" 
        
        if ahr < 0.45: 
            state_text = "ZONE L (Buy)"
            color_hex = "#28a745"
        elif 0.45 <= ahr <= 1.2: 
            state_text = "ZONE M (Accum)"
            color_hex = "#007bff"
        elif 1.2 < ahr <= 4.0: 
            state_text = "ZONE N (Hold)"
            color_hex = "#fd7e14"
        
        data_map[t] = {
            "df": df, "note": note, "price": price, 
            "ahr": ahr, "state": state_text, "color": color_hex
        }
    return data_map

# --- 3. 生成 HTML 标题 (保持原有 UI 设计) ---
def generate_header_html(ticker_name, data_dict):
    if not data_dict or data_dict['df'].empty: return "DATA LOADING FAILED"
    
    price = data_dict['price']
    ahr = data_dict['ahr']
    state = data_dict['state']
    color = data_dict['color']
    note = data_dict['note']
    
    return f"""
    <span style="font-family: 'Courier New'; font-size: 24px; font-weight: bold;">{ticker_name} DASHBOARD</span><br>
    <span style="font-size: 14px; color: #555;">{note}</span><br><br>
    
    <span style="font-size: 16px; color: #666;">CURRENT PRICE: </span>
    <span style="font-size: 20px; font-weight: bold;">${price:,.2f}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span style="font-size: 16px; color: #666;">DEVIATION INDEX: </span>
    <span style="font-size: 20px; font-weight: bold; color: {color};">{ahr:.4f}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span style="font-size: 16px; color: #666;">STATUS: </span>
    <span style="font-size: 20px; font-weight: bold; color: {color};">{state}</span>
    """

# --- 4. 绘图与执行 ---

# 加载数据 (带进度提示)
with st.spinner("Establishing secure connection to market data nodes..."):
    data_source = load_all_market_data()

# 检查是否全部失败
if data_source["BTC-USD"]["df"].empty and data_source["ETH-USD"]["df"].empty:
    st.error("❌ CRITICAL ERROR: Unable to connect to ANY market data source (Yahoo & Coingecko both failed). Please check your internet connection.")
else:
    # 即使部分失败，只要有一个成功也继续渲染
    btc_valid = not data_source["BTC-USD"]["df"].empty
    eth_valid = not data_source["ETH-USD"]["df"].empty
    
    btc_title = generate_header_html("BITCOIN", data_source["BTC-USD"]) if btc_valid else "BTC DATA ERROR"
    eth_title = generate_header_html("ETHEREUM", data_source["ETH-USD"]) if eth_valid else "ETH DATA ERROR"

    # 创建画布
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
        row_heights=[0.65, 0.35]
    )

    # --- Group 1: BTC Traces (默认显示) ---
    if btc_valid:
        btc_df = data_source["BTC-USD"]["df"]
        fig.add_trace(go.Scatter(x=btc_df.index, y=btc_df['Close'], name="BTC Price", line=dict(color="black", width=1.5), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=btc_df.index, y=btc_df['GeoMean'], name="Geo-Mean", line=dict(color="#666", width=1.5, dash='dot'), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=btc_df.index, y=btc_df['Predicted'], name="Power Law", line=dict(color="#800080", width=1.5, dash='dash'), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=btc_df.index, y=btc_df['AHR999'], name="Deviation", line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)
    else:
        # 占位符，防止索引错乱
        for _ in range(4): fig.add_trace(go.Scatter(x=[], y=[], visible=True), row=1, col=1)

    # --- Group 2: ETH Traces (默认隐藏) ---
    if eth_valid:
        eth_df = data_source["ETH-USD"]["df"]
        fig.add_trace(go.Scatter(x=eth_df.index, y=eth_df['Close'], name="ETH Price", line=dict(color="black", width=1.5), visible=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=eth_df.index, y=eth_df['GeoMean'], name="Geo-Mean", line=dict(color="#666", width=1.5, dash='dot'), visible=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=eth_df.index, y=eth_df['Predicted'], name="Regression", line=dict(color="#800080", width=1.5, dash='dash'), visible=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=eth_df.index, y=eth_df['AHR999'], name="Deviation", line=dict(color="#d35400", width=1.5), visible=False), row=2, col=1)
    else:
        for _ in range(4): fig.add_trace(go.Scatter(x=[], y=[], visible=False), row=1, col=1)

    # 参考线
    fig.add_hline(y=0.45, line_color="green", line_dash="dash", row=2, col=1, annotation_text="Buy")
    fig.add_hline(y=1.2, line_color="blue", line_dash="dot", row=2, col=1, annotation_text="Accum")
    fig.add_hline(y=4.0, line_color="red", line_dash="dash", row=2, col=1, annotation_text="Sell")

    # --- 纯 Plotly 交互按钮 ---
    updatemenus = [
        dict(
            type="buttons", direction="right", active=0, x=0, y=1.2,
            buttons=list([
                dict(label=" ₿ BTC ", method="update",
                     args=[{"visible": [True, True, True, True, False, False, False, False]},
                           {"title.text": btc_title}]),
                dict(label=" ♦ ETH ", method="update",
                     args=[{"visible": [False, False, False, False, True, True, True, True]},
                           {"title.text": eth_title}]),
            ]),
            bgcolor="white", bordercolor="black", borderwidth=1,
            font=dict(family="Courier New", size=14, color="black")
        )
    ]

    fig.update_layout(
        height=850,
        title=dict(
            text=btc_title, # 默认标题
            x=0.5, y=0.96, xanchor='center', yanchor='top'
        ),
        updatemenus=updatemenus,
        template="plotly_white",
        font=dict(family="Times New Roman", color="black"),
        margin=dict(t=160, l=60, r=40, b=80),
        xaxis=dict(
            rangeslider=dict(visible=True, borderwidth=1, bordercolor="black"),
            type="date", showgrid=True, gridcolor="#eee", linecolor="black", mirror=True
        ),
        yaxis=dict(type="log", title="Price (Log)", showgrid=True, gridcolor="#eee", linecolor="black", mirror=True),
        yaxis2=dict(title="Deviation Index", showgrid=True, gridcolor="#eee", linecolor="black", mirror=True),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    
    fig.update_shapes(dict(line_color="black"))
    
    # 唯一输出
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
