import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
from plotly.utils import PlotlyJSONEncoder # 核心修复：引入 Plotly 专用编码器

# --- 1. 页面配置 ---
st.set_page_config(page_title="Market Cycle Monitor", layout="wide")
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. 数据获取 ---
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

# --- 3. 绘图逻辑 ---
def create_chart(df_btc, df_eth):
    c_p, c_b, c_a, c_r = "#000000", "#228b22", "#4682b4", "#b22222"
    
    # 1. 创建图表结构
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08)

    # 2. 绘制 BTC (Trace 0, 1, 2)
    if not df_btc.empty:
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Close'], name="BTC Price", line=dict(color=c_p, width=1.5), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['Predicted'], name="BTC Model", line=dict(color="purple", width=1, dash='dash'), visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc.index, y=df_btc['AHR999'], name="BTC Index", line=dict(color="#d35400", width=1.5), visible=True), row=2, col=1)

    # 3. 绘制 ETH (Trace 3, 4, 5) - 影子轴
    if not df_eth.empty:
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Close'], name="ETH Price", line=dict(color=c_p, width=1.5), yaxis="y3", visible=False)) 
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['Predicted'], name="ETH Model", line=dict(color="purple", width=1, dash='dash'), yaxis="y3", visible=False))
        fig.add_trace(go.Scatter(x=df_eth.index, y=df_eth['AHR999'], name="ETH Index", line=dict(color="#d35400", width=1.5), yaxis="y4", visible=False))

    # 4. 背景区域
    for y_val, c, tx in [(0.45, c_b, "BUY"), (1.2, c_a, "ACCUM"), (4.0, c_r, "RISK")]:
        fig.add_hline(y=y_val, row=2, col=1, line_dash="dot", line_color=c, annotation_text=tx, annotation_position="top left", annotation_font=dict(color=c, size=10))

    # 5. 按钮定义
    t_btc = f"<b>BTC-USD</b>: ${df_btc['Close'].iloc[-1]:,.2f}" if not df_btc.empty else "BTC"
    t_eth = f"<b>ETH-USD</b>: ${df_eth['Close'].iloc[-1]:,.2f}" if not df_eth.empty else "ETH"

    # 注意：这里我们设置 autorange=True，让 JS 接管后的首次重绘能找到基准
    btn_btc = dict(
        label="BTC", method="update",
        args=[
            {"visible": [True, True, True, False, False, False]},
            {
                "title.text": t_btc,
                "yaxis.visible": True, "yaxis2.visible": True,
                "yaxis3.visible": False, "yaxis4.visible": False
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
                "yaxis3.visible": True, "yaxis4.visible": True
            }
        ]
    )

    # 6. 布局配置
    # 全局关闭默认 RangeSelector
    fig.update_xaxes(rangeselector=dict(visible=False), rangeslider=dict(visible=False), fixedrange=False)
    # 底部开启 RangeSlider
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05, bgcolor="#f4f4f4"), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=80, l=40, r=40, b=40),
        title=dict(text=t_btc, x=0.01, y=0.96),
        hovermode="x unified",
        showlegend=False,
        dragmode="pan", 
        
        updatemenus=[dict(
            type="buttons", direction="left", active=0, x=0.01, y=1.08,
            buttons=[btn_btc, btn_eth], bgcolor="white", bordercolor="#e0e0e0", borderwidth=1
        )],
        
        # 关键：所有 Y 轴 fixedrange=False，否则无法缩放
        yaxis=dict(domain=[0.35, 1], type="log", title="Price", fixedrange=False),
        yaxis2=dict(domain=[0, 0.30], type="log", title="Index", fixedrange=False),
        yaxis3=dict(domain=[0.35, 1], anchor="x", overlaying="y", side="left", type="log", visible=False, showgrid=False, fixedrange=False),
        yaxis4=dict(domain=[0, 0.30], anchor="x", overlaying="y2", side="left", type="log", visible=False, showgrid=False, fixedrange=False)
    )
    return fig

# --- 4. 核心组件：带 JS 自动缩放的图表渲染 ---
def st_plotly_autoscaling(fig):
    """
    使用 JavaScript 注入实现：
    1. 渲染 Plotly 图表
    2. 监听 'plotly_relayout' 事件 (缩放/平移)
    3. 计算当前 X 轴视口内的数据范围
    4. 自动更新 Y 轴范围
    """
    
    # [核心修复] 使用 PlotlyJSONEncoder 处理 Numpy/Pandas 数据类型
    plot_json = json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ margin: 0; }}
            .modebar {{ display: none !important; }}
        </style>
    </head>
    <body>
        <div id="myDiv" style="height: 700px; width: 100%;"></div>
        <script>
            var plotData = {plot_json};
            
            // 初始化图表
            Plotly.newPlot('myDiv', plotData.data, plotData.layout, {{responsive: true, displayModeBar: false, scrollZoom: true}});
            
            var graphDiv = document.getElementById('myDiv');
            var relayoutTimer;

            // 监听布局变化 (缩放/平移)
            graphDiv.on('plotly_relayout', function(eventdata){{
                // 如果事件是由 Y 轴变化触发的，忽略之，防止死循环
                if (eventdata["yaxis.range[0]"] || eventdata["yaxis3.range[0]"] || eventdata["yaxis2.range[0]"]) return;
                
                // 获取当前的 X 轴范围
                var xrange = graphDiv.layout.xaxis.range;
                
                // 如果是 Autorange (双击重置)，不需要计算，Plotly 会自动处理
                if (!xrange || eventdata["xaxis.autorange"]) return;

                // 防抖动：等待 50ms 后再计算，避免拖拽时计算过于频繁
                clearTimeout(relayoutTimer);
                relayoutTimer = setTimeout(function(){{
                    var xMin = new Date(xrange[0]).getTime();
                    var xMax = new Date(xrange[1]).getTime();
                    
                    // 判断当前显示的是 BTC (trace 0) 还是 ETH (trace 3)
                    // Plotly 中 trace 0,1,2 是 BTC; 3,4,5 是 ETH
                    var isBTC = graphDiv.data[0].visible !== false;
                    
                    // 定义我们要计算的 trace 索引和对应的 Y 轴名称
                    // 我们分别计算 Price (y/y3) 和 Index (y2/y4)
                    var priceTraceIdx = isBTC ? 0 : 3;
                    var indexTraceIdx = isBTC ? 2 : 5;
                    
                    var yPriceName = isBTC ? 'yaxis' : 'yaxis3';
                    var yIndexName = isBTC ? 'yaxis2' : 'yaxis4';
                    
                    // 辅助函数：计算局部 Min/Max
                    function getRange(traceIndex) {{
                        var xData = graphDiv.data[traceIndex].x;
                        var yData = graphDiv.data[traceIndex].y;
                        var minVal = Infinity;
                        var maxVal = -Infinity;
                        var hasData = false;

                        for (var i = 0; i < xData.length; i++) {{
                            var xVal = new Date(xData[i]).getTime();
                            if (xVal >= xMin && xVal <= xMax) {{
                                var yVal = yData[i];
                                if (yVal > 0) {{ // Log 轴不能有 0 或负数
                                    if (yVal < minVal) minVal = yVal;
                                    if (yVal > maxVal) maxVal = yVal;
                                    hasData = true;
                                }}
                            }}
                        }}
                        return hasData ? [minVal, maxVal] : null;
                    }}
                    
                    // 计算新范围
                    var priceRange = getRange(priceTraceIdx);
                    var indexRange = getRange(indexTraceIdx);
                    
                    var update = {{}};
                    
                    // 更新价格轴
                    if (priceRange) {{
                        var logMin = Math.log10(priceRange[0]);
                        var logMax = Math.log10(priceRange[1]);
                        var pad = (logMax - logMin) * 0.1; // 10% 留白
                        if (pad === 0) pad = 0.1;
                        update[yPriceName + '.range'] = [logMin - pad, logMax + pad];
                    }}
                    
                    // 更新指数轴 (可选：如果你希望下方的 Index 也自动缩放，取消注释下面代码)
                    /*
                    if (indexRange) {{
                        var logMinI = Math.log10(indexRange[0]);
                        var logMaxI = Math.log10(indexRange[1]);
                        var padI = (logMaxI - logMinI) * 0.1;
                        if (padI === 0) padI = 0.1;
                        update[yIndexName + '.range'] = [logMinI - padI, logMaxI + padI];
                    }}
                    */

                    // 执行更新
                    if (Object.keys(update).length > 0) {{
                        Plotly.relayout(graphDiv, update);
                    }}
                    
                }}, 50);
            }});
        </script>
    </body>
    </html>
    """
    # 渲染 HTML 组件
    components.html(html_code, height=710, scrolling=False)

# --- 5. 主程序 ---
st.title("Market Cycle Monitor")
with st.spinner("Syncing data..."):
    btc_df, eth_df = get_data("BTC-USD"), get_data("ETH-USD")

if not btc_df.empty:
    fig = create_chart(btc_df, eth_df)
    st_plotly_autoscaling(fig)
else:
    st.error("Data Load Failed")
