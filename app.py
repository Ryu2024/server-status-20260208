import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components # 引入组件库用于注入 JS
import json 

# --- 1. 页面配置 ---
st.set_page_config(page_title="Market Cycle Monitor", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
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
            data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
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

# --- 3. 绘图逻辑 (确保所有轴可缩放) ---
def create_chart(df_btc, df_eth):
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)

    # --- BTC (默认可见) ---
    if not df_btc.empty:
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name="BTC Price", line=dict(color=c_p, width=1.5), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Predicted'], name="BTC Model", line=dict(color="purple", width=1, dash='dash'), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['AHR999'], name="BTC Index", line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # --- ETH (影子轴, 默认隐藏) ---
    if not df_eth.empty:
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Close'], name="ETH Price", line=dict(color=c_p, width=1.5), yaxis="y3", visible=False)) 
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Predicted'], name="ETH Model", line=dict(color="purple", width=1, dash='dash'), yaxis="y3", visible=False))
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['AHR999'], name="ETH Index", line=dict(color="#d35400", width=1.5), yaxis="y4", visible=False))

    # --- 背景线 ---
    for y_val, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_position="top left", annotation_font=dict(color=c, size=10))

    # --- 按钮逻辑 ---
    t_btc = f"<b>BTC-USD</b>: ${df_btc['Close'].iloc[-1]:,.2f}" if not df_btc.empty else "BTC"
    t_eth = f"<b>ETH-USD</b>: ${df_eth['Close'].iloc[-1]:,.2f}" if not df_eth.empty else "ETH"

    # 重要修改：切换时，我们也显式开启对应轴的 autorange，方便 JS 介入
    btn_btc = dict(
        label="BTC", method="update",
        args=[
            {"visible": [True, True, True, False, False, False]},
            {
                "title.text": t_btc,
                "yaxis.visible": True, "yaxis2.visible": True,
                "yaxis3.visible": False, "yaxis4.visible": False,
                # 确保切回来时是 Auto 状态，防止被锁死
                "yaxis.autorange": True, "yaxis2.autorange": True 
            }
        ]
    )
    
    btn_eth = dict(
        label="ETH", method="update",
        args=[
            {"visible": [False, False, False, True, True, True]},
            {
                "title.text": t_eth,
                "yaxis.visible": False, "yaxis2.visible": False,
                "yaxis3.visible": True, "yaxis4.visible": True,
                "yaxis3.autorange": True, "yaxis4.autorange": True
            }
        ]
    )

    # --- 布局配置 ---
    # 彻底清除按钮，只留滑块
    fig.update_xaxes(rangeselector=dict(visible=False), rangeslider=dict(visible=False), fixedrange=False)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05, bgcolor="#f4f4f4"), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=80, l=40, r=40, b=60),
        title=dict(text=t_btc, x=0.01, y=0.96),
        hovermode="x unified",
        showlegend=False,
        dragmode="pan", # 开启平移
        
        updatemenus=[dict(
            type="buttons", direction="left", active=0, x=0.01, y=1.08,
            buttons=[btn_btc, btn_eth], bgcolor="white", bordercolor="#e0e0e0", borderwidth=1
        )],
        
        # 关键：所有 Y 轴必须 fixedrange=False 才能支持缩放（无论是手动还是 JS）
        yaxis=dict(domain=[0.35, 1], type="log", title="Price", fixedrange=False),
        yaxis2=dict(domain=[0, 0.30], type="log", title="Index", fixedrange=False),
        yaxis3=dict(domain=[0.35, 1], anchor="x", overlaying="y", side="left", type="log", visible=False, showgrid=False, fixedrange=False),
        yaxis4=dict(domain=[0, 0.30], anchor="x", overlaying="y2", side="left", type="log", visible=False, showgrid=False, fixedrange=False)
    )

    return fig

# --- 4. 终极组件：Plotly + JS AutoScale ---
def st_plotly_autoscaling(fig):
    """
    不使用 st.plotly_chart，而是直接生成 HTML。
    注入 JS 监听 'plotly_relayout' 事件，当 X 轴变化时，
    自动计算当前视口内的 Y 轴范围并更新。
    """
    plot_json = json.dumps(fig.to_dict(), cls=None)
    
    # 这段 JS 是核心：
    # 1. 监听 relayout (缩放/拖动)
    # 2. 遍历数据，找到在当前 X 轴范围内的数据点
    # 3. 计算新的 Min/Max 并 relayout Y 轴
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            .modebar {{ display: none !important; }}
        </style>
    </head>
    <body>
        <div id="myDiv" style="height: 700px; width: 100%;"></div>
        <script>
            var plotData = {plot_json};
            
            Plotly.newPlot('myDiv', plotData.data, plotData.layout, {{responsive: true, displayModeBar: false, scrollZoom: true}});
            
            var graphDiv = document.getElementById('myDiv');
            
            // 防抖动 Timer
            var relayoutTimer;

            graphDiv.on('plotly_relayout', function(eventdata){{
                // 如果是 Y 轴变化引发的，忽略，防止死循环
                if (eventdata["yaxis.range[0]"] || eventdata["yaxis3.range[0]"]) return;
                
                // 获取 X 轴范围
                var xrange = graphDiv.layout.xaxis.range;
                
                // 如果是 Autorange (双击重置)，则不需要计算
                if (!xrange || eventdata["xaxis.autorange"]) return;

                clearTimeout(relayoutTimer);
                relayoutTimer = setTimeout(function(){{
                    var xMin = new Date(xrange[0]).getTime();
                    var xMax = new Date(xrange[1]).getTime();
                    
                    // 判断当前是 BTC (trace 0 visible) 还是 ETH (trace 3 visible)
                    // Plotly trace index: 0=BTC, 3=ETH
                    var isBTC = graphDiv.data[0].visible !== false;
                    var traceIndex = isBTC ? 0 : 3;
                    var yAxisName = isBTC ? 'yaxis' : 'yaxis3';
                    
                    var xData = graphDiv.data[traceIndex].x;
                    var yData = graphDiv.data[traceIndex].y;
                    
                    var minY = Infinity;
                    var maxY = -Infinity;
                    
                    // 遍历数据寻找局部极值
                    for (var i = 0; i < xData.length; i++) {{
                        var xVal = new Date(xData[i]).getTime();
                        if (xVal >= xMin && xVal <= xMax) {{
                            var yVal = yData[i];
                            if (yVal > 0) {{ // Log axis protection
                                if (yVal < minY) minY = yVal;
                                if (yVal > maxY) maxY = yVal;
                            }}
                        }}
                    }}
                    
                    if (minY !== Infinity && maxY !== -Infinity) {{
                        // Log Scale Buffer
                        var logMin = Math.log10(minY);
                        var logMax = Math.log10(maxY);
                        var padding = (logMax - logMin) * 0.1; // 10% padding
                        
                        var update = {{}};
                        update[yAxisName + '.range'] = [logMin - padding, logMax + padding];
                        
                        Plotly.relayout(graphDiv, update);
                    }}
                }}, 100); // 100ms 延迟
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=720, scrolling=False)

# --- 5. 主程序 ---
st.title("Market Cycle Monitor")
with st.spinner("Syncing..."):
    btc_df, eth_df = get_data("BTC-USD"), get_data("ETH-USD")

if not btc_df.empty:
    fig = create_chart(btc_df, eth_df)
    
    # 使用自定义组件渲染，替代 st.plotly_chart
    # 这会启用自动缩放功能
    st_plotly_autoscaling(fig)
else:
    st.error("Load Failed")
