import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- 1. 页面基础配置 (仅保留最基础的容器) ---
st.set_page_config(page_title="Retro Crypto Dashboard", layout="wide")

# --- 2. 数据计算逻辑 (复用原逻辑，但设计为一次性获取) ---
def calculate_metrics(df, ticker):
    """核心计算逻辑"""
    if df.empty: return df, "No Data", 0, 0, "N/A"
    
    # 清洗数据
    if isinstance(df.columns, pd.MultiIndex):
        if'Close' in df.columns.get_level_values(0): df = df.xs('Close', axis=1, level=0, drop_level=True)
        else: df.columns = df.columns.droplevel(1)
    if'Close' not in df.columns:
        # 尝试修复列名
        close_cols = [c for c in df.columns if'Close' in str(c)]
        if close_cols: 
            df = df[[close_cols[0]]].copy()
            df.columns = ['Close']
        else:
            return pd.DataFrame(), "Column Error", 0, 0, "Error"

    df = df[['Close']].copy().sort_index()
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df[df['Close'] > 0].dropna()
    
    # 计算指标
    df['Log_Price'] = np.log(df['Close'])
    df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
    
    genesis_date = pd.Timestamp("2009-01-03")
    df['Days'] = (df.index - genesis_date).days
    df = df[df['Days'] > 0]

    # 预测模型
    if ticker == "BTC-USD":
        slope = 5.84
        intercept = -17.01
        log_days = np.log10(df['Days'])
        df['Predicted'] = 10 ** (slope * log_days + intercept)
        note = "Model: Power Law (Fixed)"
    else:
        # ETH 动态回归
        valid_data = df.dropna()
        if len(valid_data) > 10:
            x = np.log10(valid_data['Days'].values)
            y = np.log10(valid_data['Close'].values)
            slope, intercept, _, _, _ = linregress(x, y)
            df['Predicted'] = 10 ** (intercept + slope * np.log10(df['Days']))
            note = f"Model: Dynamic Reg (Beta {slope:.2f})"
        else:
            df['Predicted'] = np.nan
            note = "Insufficient Data"

    # AHR999 / 偏离度计算
    df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
    
    # 获取最新状态
    last_row = df.iloc[-1]
    current_price = last_row['Close']
    current_ahr = last_row['AHR999'] if not np.isnan(last_row['AHR999']) else 0
    
    return df, note, current_price, current_ahr

@st.cache_data(ttl=3600)
def fetch_all_data():
    """预加载 BTC 和 ETH 数据"""
    data_map = {}
    tickers = ["BTC-USD", "ETH-USD"]
    
    for t in tickers:
        try:
            raw_df = yf.download(t, period="max", interval="1d", progress=False)
            df, note, price, ahr = calculate_metrics(raw_df, t)
            
            # 确定状态文本和颜色
            state_text = "ZONE H (Overshoot)"
            color_hex = "#dc3545" # Red
            
            if ahr < 0.45: 
                state_text = "ZONE L (Undershoot)"
                color_hex = "#28a745" # Green
            elif 0.45 <= ahr <= 1.2: 
                state_text = "ZONE M (Accumulation)"
                color_hex = "#007bff" # Blue
            elif 1.2 < ahr <= 4.0: 
                state_text = "ZONE N (Neutral)"
                color_hex = "#fd7e14" # Orange
            
            data_map[t] = {
                "df": df,
                "note": note,
                "price": price,
                "ahr": ahr,
                "state": state_text,
                "color": color_hex
            }
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            data_map[t] = None
            
    return data_map

# --- 3. 生成 HTML 注解文本 ---
def generate_header_html(ticker_name, data_dict):
    """生成模拟 Metrics Card 的 HTML 标题"""
    if not data_dict: return "Data Error"
    
    price = data_dict['price']
    ahr = data_dict['ahr']
    state = data_dict['state']
    color = data_dict['color']
    note = data_dict['note']
    
    # 构造复古风格的 HTML (Times New Roman)
    html = f"""
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
    return html

# --- 4. 主程序逻辑 ---

data_source = fetch_all_data()

if data_source["BTC-USD"] is None:
    st.error("Failed to load market data.")
else:
    # 创建子图结构
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08, 
        row_heights=[0.65, 0.35],
        subplot_titles=("", "") # 标题通过 Annotation 动态控制
    )

    # ---------------------------------------------------------
    # 核心技巧：添加所有 Trace，但根据 Ticker 分组
    # ---------------------------------------------------------
    
    # Group 1: BTC Traces (默认 Visible = True)
    btc_data = data_source["BTC-USD"]["df"]
    # Trace 0: Price
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Close'], name="BTC Price", 
                             line=dict(color="#000000", width=1.5), visible=True), row=1, col=1)
    # Trace 1: GeoMean
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['GeoMean'], name="Geo-Mean", 
                             line=dict(color="#666666", width=1.5, dash='dot'), visible=True), row=1, col=1)
    # Trace 2: Predicted
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['Predicted'], name="Power Law", 
                             line=dict(color="#800080", width=1.5, dash='dash'), visible=True), row=1, col=1)
    # Trace 3: AHR/Deviation
    fig.add_trace(go.Scatter(x=btc_data.index, y=btc_data['AHR999'], name="Deviation Index", 
                             line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # Group 2: ETH Traces (默认 Visible = False)
    eth_data = data_source["ETH-USD"]["df"]
    # Trace 4: Price
    fig.add_trace(go.Scatter(x=eth_data.index, y=eth_data['Close'], name="ETH Price", 
                             line=dict(color="#000000", width=1.5), visible=False), row=1, col=1)
    # Trace 5: GeoMean
    fig.add_trace(go.Scatter(x=eth_data.index, y=eth_data['GeoMean'], name="Geo-Mean", 
                             line=dict(color="#666666", width=1.5, dash='dot'), visible=False), row=1, col=1)
    # Trace 6: Predicted
    fig.add_trace(go.Scatter(x=eth_data.index, y=eth_data['Predicted'], name="Regression", 
                             line=dict(color="#800080", width=1.5, dash='dash'), visible=False), row=1, col=1)
    # Trace 7: AHR/Deviation
    fig.add_trace(go.Scatter(x=eth_data.index, y=eth_data['AHR999'], name="Deviation Index", 
                             line=dict(color="#d35400", width=1.5), visible=False), row=2, col=1)

    # ---------------------------------------------------------
    # 静态参考线 (始终显示)
    # ---------------------------------------------------------
    fig.add_hline(y=0.45, line_color="green", line_dash="dash", row=2, col=1, annotation_text="Buy (0.45)")
    fig.add_hline(y=1.2, line_color="blue", line_dash="dot", row=2, col=1, annotation_text="Accum (1.2)")
    fig.add_hline(y=4.0, line_color="red", line_dash="dash", row=2, col=1, annotation_text="Sell (4.0)")

    # ---------------------------------------------------------
    # 交互核心：Update Menus (按钮)
    # ---------------------------------------------------------
    
    # 预计算两种状态下的 Title HTML
    btc_title = generate_header_html("BITCOIN", data_source["BTC-USD"])
    eth_title = generate_header_html("ETHEREUM", data_source["ETH-USD"])

    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0,
            y=1.2, # 放在顶部
            buttons=list([
                dict(label=" ₿ BTC ANALYSIS ",
                     method="update",
                     args=[{"visible": [True, True, True, True, False, False, False, False]}, # 显示前4个，隐藏后4个
                           {"title.text": btc_title}]),
                dict(label=" ♦ ETH ANALYSIS ",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, True, True]}, # 隐藏前4个，显示后4个
                           {"title.text": eth_title}]),
            ]),
            bgcolor="#ffffff",
            bordercolor="#000000",
            borderwidth=1,
            font=dict(family="Courier New", size=14, color="#000000")
        )
    ]

    # ---------------------------------------------------------
    # 布局美化 (Retro Style)
    # ---------------------------------------------------------
    fig.update_layout(
        height=800,
        title=dict(
            text=btc_title, # 默认显示 BTC
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(family="Times New Roman", size=12) # HTML 控制实际大小
        ),
        updatemenus=updatemenus,
        template="plotly_white",
        font=dict(family="Times New Roman", color="#000"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=180, l=60, r=40, b=80), # 顶部留白给 Title HTML
        
        # 范围滑块 (替代原本的 Slider)
        xaxis=dict(
            rangeslider=dict(visible=True, borderwidth=1, bordercolor="#000"),
            type="date",
            showgrid=True, gridcolor="#eee", linecolor="black", mirror=True
        ),
        xaxis2=dict(
            showgrid=True, gridcolor="#eee", linecolor="black", mirror=True
        ),
        yaxis=dict(
            type="log", title="Price (Log)", 
            showgrid=True, gridcolor="#eee", linecolor="black", mirror=True
        ),
        yaxis2=dict(
            title="Deviation", 
            showgrid=True, gridcolor="#eee", linecolor="black", mirror=True
        ),
        # 图例配置
        legend=dict(
            orientation="h", y=-0.2, x=0.5, xanchor="center",
            bordercolor="black", borderwidth=1, bgcolor="white"
        )
    )

    # 绘制边框效果 (Shapes)
    fig.update_shapes(dict(line_color="black"))

    # ---------------------------------------------------------
    # 渲染
    # ---------------------------------------------------------
    # config 中开启 responsive，去掉 plotly logo
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})
