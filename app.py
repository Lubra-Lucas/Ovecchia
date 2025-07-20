import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LUBRA Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📈 LUBRA Trading Analysis Dashboard")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Trading Parameters")

# Input fields in sidebar
with st.sidebar:
    st.subheader("Asset Configuration")
    
    # Symbol input with examples
    symbol = st.text_input(
        "Asset Symbol",
        value="BTC-USD",
        help="Examples: BTC-USD, PETR4.SA, AAPL, EURUSD=X"
    ).strip()
    
    # Date range selection
    st.subheader("Date Range")
    
    # Default date range (last 30 days)
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=30)
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=default_end
    )
    
    end_date = st.date_input(
        "End Date",
        value=default_end,
        min_value=start_date,
        max_value=default_end
    )
    
    # Interval selection
    st.subheader("Time Interval")
    interval_options = {
        "1 minute": "1m",
        "5 minutes": "5m",
        "15 minutes": "15m",
        "30 minutes": "30m",
        "1 hour": "1h",
        "1 day": "1d",
        "1 week": "1wk",
        "1 month": "1mo"
    }
    
    interval_display = st.selectbox(
        "Select Interval",
        list(interval_options.keys()),
        index=5  # Default to "1 day"
    )
    interval = interval_options[interval_display]
    
    # Confirmation candles parameter
    st.subheader("Signal Confirmation")
    confirm_candles = st.number_input(
        "Confirmation Candles",
        min_value=0,
        max_value=5,
        value=0,
        help="Number of consecutive candles with same signal needed for validation"
    )
    
    # Stop loss selection
    st.subheader("Stop Loss Display")
    stop_options = {
        "Stop Justo (1.5x ATR)": "Stop_Justo",
        "Stop Balanceado (2.0x ATR)": "Stop_Balanceado", 
        "Stop Largo (3.0x ATR)": "Stop_Largo"
    }
    
    selected_stop_display = st.selectbox(
        "Select Stop Loss Type",
        list(stop_options.keys()),
        index=0
    )
    selected_stop = stop_options[selected_stop_display]
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Main content area
if analyze_button:
    if not symbol:
        st.error("Please enter a valid asset symbol.")
        st.stop()
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Fetch data
        status_text.text("Fetching market data...")
        progress_bar.progress(20)
        
        # Convert dates to strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Download data
        df = yf.download(symbol, start=start_str, end=end_str, interval=interval)
        
        if df is None or df.empty:
            st.error(f"No data found for symbol '{symbol}' in the specified date range.")
            st.stop()
        
        # Handle multi-level columns if present
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df = df.xs(symbol, level='Ticker', axis=1, drop_level=True)
        
        progress_bar.progress(40)
        status_text.text("Processing technical indicators...")
        
        # Data preprocessing
        symbol_label = symbol.replace("=X", "")
        df.reset_index(inplace=True)
        
        # Standardize column names
        column_mapping = {
            "Datetime": "time", 
            "Date": "time", 
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close",
            "Volume": "volume"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure we have the required columns
        required_columns = ['time', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required data columns: {missing_columns}")
            st.stop()
        
        progress_bar.progress(60)
        
        # Calculate technical indicators
        # Moving averages
        df['SMA_60'] = df['close'].rolling(window=60).mean()
        df['SMA_70'] = df['close'].rolling(window=70).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # RSL calculation
        df['RSL_20'] = df['close'] / df['SMA_20']
        
        progress_bar.progress(80)
        status_text.text("Generating trading signals...")
        
        # Signal generation
        df['Signal'] = 'Stay Out'
        for i in range(1, len(df)):
            rsi_up = df['RSI_14'].iloc[i] > df['RSI_14'].iloc[i-1]
            rsi_down = df['RSI_14'].iloc[i] < df['RSI_14'].iloc[i-1]
            rsl = df['RSL_20'].iloc[i]
            rsl_prev = df['RSL_20'].iloc[i-1]
            
            rsl_buy = (rsl > 1 and rsl > rsl_prev) or (rsl < 1 and rsl > rsl_prev)
            rsl_sell = (rsl > 1 and rsl < rsl_prev) or (rsl < 1 and rsl < rsl_prev)
            
            if (
                df['close'].iloc[i] > df['SMA_60'].iloc[i]
                and df['close'].iloc[i] > df['SMA_70'].iloc[i]
                and rsi_up and rsl_buy
            ):
                df.at[i, 'Signal'] = 'Buy'
            elif (
                df['close'].iloc[i] < df['SMA_60'].iloc[i]
                and rsi_down and rsl_sell
            ):
                df.at[i, 'Signal'] = 'Sell'
        
        # State persistence with confirmation filter
        df['Estado'] = 'Stay Out'
        for i in range(confirm_candles, len(df)):
            last_signals = df['Signal'].iloc[i - confirm_candles:i]
            current_signal = df['Signal'].iloc[i]
            
            if all(last_signals == current_signal) and current_signal != 'Stay Out':
                df.loc[df.index[i], 'Estado'] = current_signal
            else:
                df.loc[df.index[i], 'Estado'] = df['Estado'].iloc[i - 1]
        
        # ATR and Stop Loss calculations
        df['prior_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prior_close'])
        df['tr3'] = abs(df['low'] - df['prior_close'])
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Initialize stop loss levels
        df['Stop_Justo'] = np.nan
        df['Stop_Balanceado'] = np.nan
        df['Stop_Largo'] = np.nan
        
        # ATR factors for each stop type
        fatores = {'Stop_Justo': 1.5, 'Stop_Balanceado': 2.0, 'Stop_Largo': 3.0}
        
        for i in range(1, len(df)):
            estado = df['Estado'].iloc[i]
            close = df['close'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            for stop_tipo, fator in fatores.items():
                stop_anterior = df[stop_tipo].iloc[i - 1]
                if estado == 'Buy':
                    stop_atual = close - fator * atr
                    df.loc[df.index[i], stop_tipo] = max(stop_anterior, stop_atual) if pd.notna(stop_anterior) else stop_atual
                elif estado == 'Sell':
                    stop_atual = close + fator * atr
                    df.loc[df.index[i], stop_tipo] = min(stop_anterior, stop_atual) if pd.notna(stop_anterior) else stop_atual
        
        # Color coding and indicators
        df['Color'] = 'black'
        df.loc[df['Estado'] == 'Buy', 'Color'] = 'blue'
        df.loc[df['Estado'] == 'Sell', 'Color'] = 'red'
        # Create indicator mapping
        estado_mapping = {'Buy': 1, 'Sell': 0, 'Stay Out': 0.5}
        df['Indicator'] = df['Estado'].apply(lambda x: estado_mapping.get(x, 0.5))
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"✅ Analysis completed for {symbol_label}")
        
        # Current status display
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df['close'].iloc[-1]
        current_signal = df['Estado'].iloc[-1]
        current_rsi = df['RSI_14'].iloc[-1]
        current_rsl = df['RSL_20'].iloc[-1]
        
        with col1:
            st.metric("Current Price", f"{current_price:.2f}")
        
        with col2:
            signal_color = "🔵" if current_signal == "Buy" else "🔴" if current_signal == "Sell" else "⚫"
            st.metric("Current Signal", f"{signal_color} {current_signal}")
        
        with col3:
            st.metric("RSI (14)", f"{current_rsi:.2f}")
        
        with col4:
            st.metric("RSL (20)", f"{current_rsl:.3f}")
        
        st.markdown("---")
        
        # Create the interactive chart
        titulo_grafico = f"LUBRA TRADING - {symbol_label} - Timeframe: {interval.upper()}"
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=("Price Chart with Signals", "Signal Indicator")
        )
        
        # Add price line with color coding
        for i in range(len(df) - 1):
            fig.add_trace(go.Scatter(
                x=df['time'][i:i+2],
                y=df['close'][i:i+2],
                mode="lines",
                line=dict(color=df['Color'][i], width=2),
                showlegend=False,
                hoverinfo="skip"
            ), row=1, col=1)
        
        # Add invisible trace for hover info
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['close'],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            name='Price',
            hovertemplate="<b>Price:</b> %{y:.2f}<br><b>Time:</b> %{x}<extra></extra>",
            showlegend=False
        ), row=1, col=1)
        
        # Add selected stop loss trace
        stop_colors = {
            "Stop_Justo": "orange",
            "Stop_Balanceado": "gray", 
            "Stop_Largo": "green"
        }
        
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[selected_stop],
            mode="lines", name=selected_stop_display,
            line=dict(color=stop_colors[selected_stop], width=2, dash="dot"),
            hovertemplate=f"<b>{selected_stop_display}:</b> %{{y:.2f}}<extra></extra>"
        ), row=1, col=1)
        
        # Add signal indicator
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['Indicator'],
            mode="lines+markers",
            name="Signal Indicator",
            line=dict(color="purple", width=2),
            marker=dict(size=4),
            showlegend=False
        ), row=2, col=1)
        
        # Add legend items
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='blue', width=2),
            name='Buy Signal'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='red', width=2),
            name='Sell Signal'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='black', width=2),
            name='Stay Out'
        ), row=1, col=1)
        
        # Add reference line for signal indicator
        fig.add_shape(
            type="line",
            x0=df['time'].iloc[0],
            x1=df['time'].iloc[-1],
            y0=0.5,
            y1=0.5,
            line=dict(color="black", width=1, dash="dash"),
            xref="x", yref="y2"
        )
        
        # Update layout
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 0.5, 1], 
                        ticktext=['Sell', 'Stay Out', 'Buy'], row=2, col=1)
        fig.update_xaxes(showgrid=False, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(text=titulo_grafico, x=0.5, font=dict(size=18)),
            template="plotly_white",
            hovermode="x unified",
            height=700
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical analysis summary
        st.subheader("📊 Technical Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Moving Averages:**")
            st.write(f"• SMA 20: {df['SMA_20'].iloc[-1]:.2f}")
            st.write(f"• SMA 60: {df['SMA_60'].iloc[-1]:.2f}")
            st.write(f"• SMA 70: {df['SMA_70'].iloc[-1]:.2f}")
            
            st.write("**Stop Loss Levels:**")
            st.write(f"• Stop Justo: {df['Stop_Justo'].iloc[-1]:.2f}")
            st.write(f"• Stop Balanceado: {df['Stop_Balanceado'].iloc[-1]:.2f}")
            st.write(f"• Stop Largo: {df['Stop_Largo'].iloc[-1]:.2f}")
        
        with col2:
            st.write("**Technical Indicators:**")
            st.write(f"• RSI (14): {current_rsi:.2f}")
            st.write(f"• RSL (20): {current_rsl:.3f}")
            st.write(f"• ATR (14): {df['ATR'].iloc[-1]:.4f}")
            
            # Signal statistics
            buy_signals = (df['Estado'] == 'Buy').sum()
            sell_signals = (df['Estado'] == 'Sell').sum()
            stay_out = (df['Estado'] == 'Stay Out').sum()
            
            st.write("**Signal Distribution:**")
            st.write(f"• Buy signals: {buy_signals}")
            st.write(f"• Sell signals: {sell_signals}")
            st.write(f"• Stay out: {stay_out}")
        
        # Data table option
        if st.checkbox("Show Raw Data"):
            st.subheader("📋 Raw Data")
            display_columns = ['time', 'open', 'high', 'low', 'close', 'SMA_20', 'SMA_60', 'SMA_70', 
                             'RSI_14', 'RSL_20', 'Signal', 'Estado', 'ATR']
            available_columns = [col for col in display_columns if col in df.columns]
            st.dataframe(df[available_columns].tail(50), use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.write("Please check your inputs and try again.")

else:
    # Initial state - show instructions
    st.info("👈 Configure your trading parameters in the sidebar and click 'Analyze' to start the analysis.")
    
    st.subheader("📋 How to Use")
    st.write("""
    1. **Enter Asset Symbol**: Use standard ticker symbols (e.g., BTC-USD, AAPL, PETR4.SA)
    2. **Set Date Range**: Choose your analysis period
    3. **Select Time Interval**: Pick the timeframe for your analysis
    4. **Configure Confirmation**: Set how many consecutive signals are needed for validation
    5. **Click Analyze**: Generate your trading analysis
    """)
    
    st.subheader("🔍 Features")
    st.write("""
    - **Technical Indicators**: SMA (20, 60, 70), RSI (14), RSL (20), ATR (14)
    - **Trading Signals**: Buy, Sell, Stay Out with confirmation logic
    - **Stop Loss Levels**: Three different ATR-based stop levels
    - **Interactive Charts**: Zoom, pan, and hover for detailed information
    - **Real-time Data**: Powered by Yahoo Finance API
    """)

# Footer
st.markdown("---")
st.markdown("*LUBRA Trading Analysis Dashboard - For educational purposes only. Not financial advice.*")
