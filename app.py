import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def display_returns_section(returns_data, criteria_name):
    """Helper function to display returns section"""
    if not returns_data.empty:
        # Get last 10 returns
        last_returns = returns_data.tail(20).copy()
        last_returns = last_returns.sort_values('exit_time', ascending=False)
        
        # Create columns for returns display
        for idx, row in last_returns.iterrows():
            col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1.5])
            
            # Color coding for returns
            return_color = "🟢" if row['return_pct'] > 0 else "🔴"
            signal_icon = "🔵" if row['signal'] == 'Buy' else "🔴"
            
            with col1:
                st.write(f"{signal_icon}")
            
            with col2:
                st.write(f"**{row['signal']}**")
            
            with col3:
                st.write(f"Entrada: {row['entry_price']:.2f}")
                st.write(f"Saída: {row['exit_price']:.2f}")
            
            with col4:
                entry_date = row['entry_time'].strftime('%d/%m/%Y %H:%M') if hasattr(row['entry_time'], 'strftime') else str(row['entry_time'])
                exit_date = row['exit_time'].strftime('%d/%m/%Y %H:%M') if hasattr(row['exit_time'], 'strftime') else str(row['exit_time'])
                st.write(f"Entrada: {entry_date}")
                st.write(f"Saída: {exit_date}")
            
            with col5:
                st.write(f"{return_color} **{row['return_pct']:.2f}%**")
                # Show exit reason for custom criteria
                if 'exit_reason' in row and pd.notna(row['exit_reason']):
                    st.caption(f"Saída: {row['exit_reason']}")
            
            st.markdown("---")
        
        # Summary statistics
        total_trades = len(returns_data)
        profitable_trades = len(returns_data[returns_data['return_pct'] > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_return = returns_data['return_pct'].mean() if not returns_data.empty else 0
        total_return = returns_data['return_pct'].sum() if not returns_data.empty else 0
        
        # Find best and worst trades
        best_trade = returns_data.loc[returns_data['return_pct'].idxmax()] if not returns_data.empty else None
        worst_trade = returns_data.loc[returns_data['return_pct'].idxmin()] if not returns_data.empty else None
        
        # Display main statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Operações", total_trades)
        with col2:
            st.metric("Operações Lucrativas", profitable_trades)
        with col3:
            st.metric("Taxa de Acerto", f"{win_rate:.1f}%")
        with col4:
            st.metric("Retorno Médio", f"{avg_return:.2f}%")
        
        # Display additional statistics
        st.markdown("### 📊 Estatísticas Detalhadas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            return_color = "🟢" if total_return >= 0 else "🔴"
            st.metric("Retorno Total do Modelo", f"{return_color} {total_return:.2f}%")
        
        with col2:
            if best_trade is not None:
                best_date = best_trade['exit_time'].strftime('%d/%m/%Y') if hasattr(best_trade['exit_time'], 'strftime') else str(best_trade['exit_time'])
                st.metric("Maior Ganho", f"🟢 {best_trade['return_pct']:.2f}%")
                st.caption(f"Data: {best_date}")
            else:
                st.metric("Maior Ganho", "N/A")
        
        with col3:
            if worst_trade is not None:
                worst_date = worst_trade['exit_time'].strftime('%d/%m/%Y') if hasattr(worst_trade['exit_time'], 'strftime') else str(worst_trade['exit_time'])
                st.metric("Maior Perda", f"🔴 {worst_trade['return_pct']:.2f}%")
                st.caption(f"Data: {worst_date}")
            else:
                st.metric("Maior Perda", "N/A")

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
        "2 minutes": "2m",
        "5 minutes": "5m",
        "15 minutes": "15m",
        "30 minutes": "30m",
        "60 minutes": "60m",
        "90 minutes": "90m",
        "1 hour": "1h",
        "4 hours": "4h",
        "1 day": "1d",
        "5 days": "5d",
        "1 week": "1wk",
        "1 month": "1mo",
        "3 months": "3mo"
    }
    
    interval_display = st.selectbox(
        "Select Interval",
        list(interval_options.keys()),
        index=9  # Default to "1 day"
    )
    interval = interval_options[interval_display]
    
    # Data limitations info
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h"]:
        st.info(
            "ℹ️ **Data Limitation:** Intraday data (minute/hour intervals) "
            "is limited to the last 60-730 days depending on the interval. "
            "For historical analysis beyond this period, use daily intervals."
        )
    
    # Confirmation candles parameter
    st.subheader("Signal Confirmation")
    confirm_candles = st.number_input(
        "Confirmation Candles",
        min_value=0,
        max_value=5,
        value=0,
        help="Number of consecutive candles with same signal needed for validation"
    )
    
    # Moving averages configuration
    st.subheader("Configuração de Médias Móveis")
    col1, col2 = st.columns(2)
    
    with col1:
        sma_short = st.number_input(
            "Média Móvel Curta",
            min_value=5,
            max_value=200,
            value=60,
            step=5,
            help="Primeira condição para sinais de compra"
        )
    
    with col2:
        sma_long = st.number_input(
            "Média Móvel Longa", 
            min_value=10,
            max_value=300,
            value=70,
            step=5,
            help="Segunda condição para sinais de compra"
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
    
    # Exit criteria configuration
    st.subheader("Critérios de Saída Personalizados")
    
    exit_criteria = st.selectbox(
        "Tipo de Saída",
        ["Mudança de Estado (Padrão)", "Stop Loss", "Alvo Fixo", "Tempo"],
        index=0,
        help="Escolha como calcular a saída das posições"
    )
    
    # Additional parameters based on exit criteria
    exit_params = {}
    
    if exit_criteria == "Stop Loss":
        exit_params['stop_type'] = st.selectbox(
            "Tipo de Stop",
            ["Stop Justo", "Stop Balanceado", "Stop Largo"]
        )
    elif exit_criteria == "Alvo Fixo":
        col1, col2 = st.columns(2)
        with col1:
            exit_params['target_pct'] = st.number_input(
                "Alvo Percentual (%)",
                min_value=0.1,
                max_value=50.0,
                value=3.0,
                step=0.1,
                help="Percentual de ganho desejado"
            )
        with col2:
            exit_params['stop_loss_pct'] = st.number_input(
                "Stop Loss Limite (%)",
                min_value=0.1,
                max_value=20.0,
                value=2.0,
                step=0.1,
                help="Máximo percentual de perda aceito"
            )
    elif exit_criteria == "Tempo":
        exit_params['time_candles'] = st.number_input(
            "Candles após entrada",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            help="Número de candles após a entrada para sair da posição"
        )
    
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
        # Moving averages (customizable)
        df[f'SMA_{sma_short}'] = df['close'].rolling(window=sma_short).mean()
        df[f'SMA_{sma_long}'] = df['close'].rolling(window=sma_long).mean()
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
                df['close'].iloc[i] > df[f'SMA_{sma_short}'].iloc[i]
                and df['close'].iloc[i] > df[f'SMA_{sma_long}'].iloc[i]
                and rsi_up and rsl_buy
            ):
                df.at[i, 'Signal'] = 'Buy'
            elif (
                df['close'].iloc[i] < df[f'SMA_{sma_short}'].iloc[i]
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
        
        # Calculate returns based on signal changes
        def calculate_signal_returns(df):
            returns_data = []
            current_signal = None
            entry_price = None
            entry_time = None
            
            for i in range(len(df)):
                estado = df['Estado'].iloc[i]
                price = df['close'].iloc[i]
                time = df['time'].iloc[i]
                
                if estado != current_signal and estado != 'Stay Out':
                    if current_signal is not None and entry_price is not None:
                        # Calculate return when signal changes
                        if current_signal == 'Buy':
                            # Exit from buy position
                            return_pct = ((price - entry_price) / entry_price) * 100
                        else:  # current_signal == 'Sell'
                            # Exit from sell position (short)
                            return_pct = ((entry_price - price) / entry_price) * 100
                        
                        returns_data.append({
                            'signal': current_signal,
                            'entry_time': entry_time,
                            'exit_time': time,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'return_pct': return_pct
                        })
                    
                    # Start new position
                    current_signal = estado
                    entry_price = price
                    entry_time = time
                elif estado == 'Stay Out' and current_signal is not None:
                    # Exit position to stay out
                    if current_signal == 'Buy':
                        return_pct = ((price - entry_price) / entry_price) * 100
                    else:  # current_signal == 'Sell'
                        return_pct = ((entry_price - price) / entry_price) * 100
                    
                    returns_data.append({
                        'signal': current_signal,
                        'entry_time': entry_time,
                        'exit_time': time,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'return_pct': return_pct
                    })
                    
                    current_signal = None
                    entry_price = None
                    entry_time = None
            
            return pd.DataFrame(returns_data)
        
        returns_df = calculate_signal_returns(df)
        
        # Calculate custom exit criteria returns
        def calculate_custom_exit_returns(df, exit_criteria, exit_params):
            if exit_criteria == "Mudança de Estado (Padrão)":
                return returns_df
            
            custom_returns = []
            current_signal = None
            entry_price = None
            entry_time = None
            entry_index = None
            
            for i in range(len(df)):
                estado = df['Estado'].iloc[i]
                price = df['close'].iloc[i]
                time = df['time'].iloc[i]
                
                # Check if we have an active position
                if current_signal is not None and entry_price is not None and entry_index is not None:
                    
                    # 1. First check for state change (highest priority)
                    if estado != current_signal:
                        # State changed - close current position immediately
                        if current_signal == 'Buy':
                            return_pct = ((price - entry_price) / entry_price) * 100
                        else:  # Sell
                            return_pct = ((entry_price - price) / entry_price) * 100
                        
                        custom_returns.append({
                            'signal': current_signal,
                            'entry_time': entry_time,
                            'exit_time': time,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'return_pct': return_pct,
                            'exit_reason': 'Mudança de Estado'
                        })
                        
                        # Start new position if new state is not Stay Out
                        if estado != 'Stay Out':
                            current_signal = estado
                            entry_price = price
                            entry_time = time
                            entry_index = i
                        else:
                            current_signal = None
                            entry_price = None
                            entry_time = None
                            entry_index = None
                        continue
                    
                    # 2. Check custom exit criteria (only if no state change)
                    exit_price, exit_time, exit_reason = calculate_exit(
                        df, entry_index, i, current_signal, entry_price, exit_criteria, exit_params
                    )
                    
                    if exit_price is not None:
                        if current_signal == 'Buy':
                            return_pct = ((exit_price - entry_price) / entry_price) * 100
                        else:  # Sell
                            return_pct = ((entry_price - exit_price) / entry_price) * 100
                        
                        custom_returns.append({
                            'signal': current_signal,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'exit_reason': exit_reason
                        })
                        
                        # Position closed by custom criteria - no new position until next signal
                        current_signal = None
                        entry_price = None
                        entry_time = None
                        entry_index = None
                        continue
                
                # Entry logic - start new position when state changes to Buy/Sell (and no active position)
                if current_signal is None and estado in ['Buy', 'Sell']:
                    current_signal = estado
                    entry_price = price
                    entry_time = time
                    entry_index = i
            
            return pd.DataFrame(custom_returns)
        
        def calculate_exit(df, entry_idx, current_idx, signal, entry_price, criteria, params):
            """Calculate exit price based on selected criteria - doesn't check state change as main loop handles it"""
            
            if criteria == "Stop Loss":
                stop_col = params['stop_type'].replace(' ', '_')
                for i in range(entry_idx + 1, min(current_idx + 1, len(df))):
                    stop_price = df[stop_col].iloc[i]
                    current_price = df['close'].iloc[i]
                    
                    if signal == 'Buy' and current_price <= stop_price:
                        return stop_price, df['time'].iloc[i], f"Stop {params['stop_type']}"
                    elif signal == 'Sell' and current_price >= stop_price:
                        return stop_price, df['time'].iloc[i], f"Stop {params['stop_type']}"
            
            elif criteria == "Alvo Fixo":
                target_pct = params['target_pct'] / 100
                stop_loss_pct = params['stop_loss_pct'] / 100
                
                if signal == 'Buy':
                    target_price = entry_price * (1 + target_pct)
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                else:
                    target_price = entry_price * (1 - target_pct)
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                
                for i in range(entry_idx + 1, min(current_idx + 1, len(df))):
                    current_price = df['close'].iloc[i]
                    
                    if signal == 'Buy':
                        if current_price >= target_price:
                            return target_price, df['time'].iloc[i], f"Alvo {params['target_pct']}%"
                        elif current_price <= stop_loss_price:
                            return stop_loss_price, df['time'].iloc[i], f"Stop Loss {params['stop_loss_pct']}%"
                    else:  # Sell
                        if current_price <= target_price:
                            return target_price, df['time'].iloc[i], f"Alvo {params['target_pct']}%"
                        elif current_price >= stop_loss_price:
                            return stop_loss_price, df['time'].iloc[i], f"Stop Loss {params['stop_loss_pct']}%"
            
            elif criteria == "Tempo":
                target_candles = params['time_candles']
                target_idx = entry_idx + target_candles
                
                if target_idx < len(df) and target_idx <= current_idx:
                    return df['close'].iloc[target_idx], df['time'].iloc[target_idx], f"Tempo {target_candles} candles"
            
            return None, None, None
        
        custom_returns_df = calculate_custom_exit_returns(df, exit_criteria, exit_params)
        
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
        
        # Returns Analysis Section
        st.subheader("📈 Análise de Retornos")
        
        # Create tabs for different return calculations
        tab1, tab2 = st.tabs(["📊 Mudança de Estado", f"🎯 {exit_criteria}"])
        
        with tab1:
            st.write("**Retornos baseados na mudança natural do estado dos sinais**")
            if not returns_df.empty:
                display_returns_section(returns_df, "Mudança de Estado")
            else:
                st.info("Nenhuma operação completa encontrada no período analisado.")
        
        with tab2:
            st.write(f"**Retornos baseados no critério: {exit_criteria}**")
            if not custom_returns_df.empty:
                display_returns_section(custom_returns_df, exit_criteria)
            else:
                st.info("Nenhuma operação completa encontrada com este critério no período analisado.")
        
        st.markdown("---")
        
        # Technical analysis summary
        st.subheader("📊 Resumo da Análise Técnica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Moving Averages:**")
            st.write(f"• SMA 20: {df['SMA_20'].iloc[-1]:.2f}")
            st.write(f"• SMA {sma_short}: {df[f'SMA_{sma_short}'].iloc[-1]:.2f}")
            st.write(f"• SMA {sma_long}: {df[f'SMA_{sma_long}'].iloc[-1]:.2f}")
            
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
