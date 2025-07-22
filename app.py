
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
        # Get last 20 returns
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

def calculate_returns(df, exit_criteria, exit_params, trading_direction, exit_on_state_change):
    """Calculate returns based on signals and exit criteria"""
    returns = []
    in_position = False
    entry_price = 0
    entry_time = None
    entry_signal = None
    
    for i in range(len(df)):
        current_signal = df['Signal'].iloc[i]
        current_price = df['close'].iloc[i]
        current_time = df['time'].iloc[i]
        
        # Entry conditions
        if not in_position:
            should_enter = False
            
            if trading_direction == "Ambos (Compra e Venda)":
                should_enter = current_signal in ['Buy', 'Sell']
            elif trading_direction == "Apenas Comprado":
                should_enter = current_signal == 'Buy'
            elif trading_direction == "Apenas Vendido":
                should_enter = current_signal == 'Sell'
            
            if should_enter:
                in_position = True
                entry_price = current_price
                entry_time = current_time
                entry_signal = current_signal
        
        # Exit conditions
        elif in_position:
            should_exit = False
            exit_reason = ""
            
            # State change exit
            if exit_on_state_change and current_signal != entry_signal and current_signal != 'Stay Out':
                should_exit = True
                exit_reason = "Mudança de Estado"
            
            # Signal exit (opposite signal)
            elif current_signal != entry_signal and current_signal in ['Buy', 'Sell']:
                should_exit = True
                exit_reason = "Sinal Oposto"
            
            if should_exit:
                # Calculate return
                if entry_signal == 'Buy':
                    return_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # Sell
                    return_pct = ((entry_price - current_price) / entry_price) * 100
                
                returns.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'signal': entry_signal,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return_pct': return_pct,
                    'exit_reason': exit_reason
                })
                
                in_position = False
    
    return pd.DataFrame(returns)

def create_chart(df, symbol, sma_short, sma_long):
    """Create the main price chart with indicators and signals"""
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} - Preço e Sinais', 'RSI', 'RSL'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Price chart with candlesticks
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Preço",
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df[f'SMA_{sma_short}'],
        mode='lines',
        name=f'SMA {sma_short}',
        line=dict(color='blue', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df[f'SMA_{sma_long}'],
        mode='lines',
        name=f'SMA {sma_long}',
        line=dict(color='orange', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(color='purple', width=1)
    ), row=1, col=1)

    # Buy and Sell signals
    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['time'],
            y=buy_signals['close'],
            mode='markers',
            name='Compra',
            marker=dict(color='green', symbol='triangle-up', size=10)
        ), row=1, col=1)

    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['time'],
            y=sell_signals['close'],
            mode='markers',
            name='Venda',
            marker=dict(color='red', symbol='triangle-down', size=10)
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['RSI_14'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=2)
    ), row=2, col=1)

    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)

    # RSL
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['RSL_20'],
        mode='lines',
        name='RSL',
        line=dict(color='orange', width=2)
    ), row=3, col=1)

    # RSL level
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f"Análise Técnica - {symbol}",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="RSL", row=3, col=1)

    return fig

# Page configuration
st.set_page_config(
    page_title="OVECCHIA TRADING - MODELO QUANT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("OVECCHIA TRADING - MODELO QUANT")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Parâmetros de Análise")

# Input fields in sidebar
with st.sidebar:
    st.subheader("Modo de Análise")

    analysis_mode = st.radio(
        "Escolha o tipo de análise:",
        ["Ativo Individual", "Screening de Múltiplos Ativos"]
    )

    if analysis_mode == "Ativo Individual":
        st.subheader("Configuração de Ativo")
        symbol = st.text_input(
            "Ticker",
            value="BTC-USD",
            help="Examples: BTC-USD, PETR4.SA, AAPL, EURUSD=X"
        ).strip()

    else:  # Screening mode
        st.subheader("Lista de Ativos para Screening")

        preset_lists = {
            "Criptomoedas Top 10": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD"],
            "Ações Brasileiras": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "MGLU3.SA", "ABEV3.SA", "JBSS3.SA", "WEGE3.SA", "RENT3.SA", "LREN3.SA"],
            "Ações Americanas": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "BABA"],
            "Pares de Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X"]
        }

        selected_preset = st.selectbox(
            "Lista Pré-definida:",
            ["Customizada"] + list(preset_lists.keys())
        )

        if selected_preset != "Customizada":
            symbols_list = preset_lists[selected_preset]
            st.info(f"Selecionados: {', '.join(symbols_list)}")
        else:
            symbols_input = st.text_area(
                "Digite os tickers (um por linha):",
                value="BTC-USD\nETH-USD\nPETR4.SA\nAAPL",
                help="Digite um ticker por linha"
            )
            symbols_list = [s.strip() for s in symbols_input.split('\n') if s.strip()]

        st.write(f"**{len(symbols_list)} ativos selecionados para screening**")

    # Date range selection
    st.subheader("Date Range")

    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=30)

    start_date = st.date_input(
        "Data Inicial",
        value=default_start,
        max_value=default_end
    )

    end_date = st.date_input(
        "Data Final",
        value=default_end,
        min_value=start_date,
        max_value=default_end
    )

    # Interval selection
    st.subheader("Intervalo de Tempo (Timeframe)")
    interval_options = {
        "1 minute": "1m", "2 minutes": "2m", "5 minutes": "5m", "15 minutes": "15m",
        "30 minutes": "30m", "60 minutes": "60m", "90 minutes": "90m", "1 hour": "1h",
        "4 hours": "4h", "1 day": "1d", "5 days": "5d", "1 week": "1wk",
        "1 month": "1mo", "3 months": "3mo"
    }

    interval_display = st.selectbox(
        "Selecione o Intervalo",
        list(interval_options.keys()),
        index=9  # Default to "1 day"
    )
    interval = interval_options[interval_display]

    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h"]:
        st.info("ℹ️ Dados intraday são limitados aos últimos 60-730 dias.")

    # Moving averages configuration
    st.subheader("Configuração de Médias Móveis")
    col1, col2 = st.columns(2)

    with col1:
        sma_short = st.number_input(
            "Média Móvel Curta",
            min_value=5, max_value=200, value=60, step=5
        )

    with col2:
        sma_long = st.number_input(
            "Média Móvel Longa", 
            min_value=10, max_value=300, value=70, step=5
        )

    # Trading direction configuration
    st.subheader("Direção de Operação")
    trading_direction = st.selectbox(
        "Escolha a direção das operações:",
        ["Ambos (Compra e Venda)", "Apenas Comprado", "Apenas Vendido"],
        index=0
    )

    # Exit criteria configuration
    st.subheader("Critérios de Saída")
    exit_criteria = st.selectbox(
        "Tipo de Saída",
        ["Mudança de Estado", "Stop Loss", "Alvo Fixo"],
        index=0
    )

    exit_on_state_change = st.checkbox(
        "🚪 Sair com Mudança de Estado",
        value=True
    )

    # Additional parameters
    exit_params = {}
    if exit_criteria == "Alvo Fixo":
        col1, col2 = st.columns(2)
        with col1:
            exit_params['target_pct'] = st.number_input(
                "Alvo Percentual (%)", min_value=0.1, max_value=50.0, value=3.0, step=0.1
            )
        with col2:
            exit_params['stop_loss_pct'] = st.number_input(
                "Stop Loss Limite (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1
            )

    # Analyze button
    analyze_button = st.button("🔍 Analisar", type="primary", use_container_width=True)

# Main content area
if analyze_button:
    if analysis_mode == "Ativo Individual":
        if not symbol:
            st.error("Por favor entre com um ticker válido.")
            st.stop()
        symbols_to_analyze = [symbol]
    else:
        if not symbols_list:
            st.error("Por favor selecione pelo menos um ativo para screening.")
            st.stop()
        symbols_to_analyze = symbols_list

    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        if analysis_mode == "Screening de Múltiplos Ativos":
            # Screening mode
            screening_results = []
            total_symbols = len(symbols_to_analyze)

            for idx, current_symbol in enumerate(symbols_to_analyze):
                status_text.text(f"Analisando {current_symbol} ({idx+1}/{total_symbols})...")
                progress_bar.progress(int((idx / total_symbols) * 100))

                try:
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")

                    df_temp = yf.download(current_symbol, start=start_str, end=end_str, interval=interval)

                    if df_temp is None or df_temp.empty:
                        screening_results.append({
                            'symbol': current_symbol, 'status': 'Erro - Sem dados',
                            'current_state': 'N/A', 'previous_state': 'N/A',
                            'state_change': False, 'current_price': 'N/A'
                        })
                        continue

                    if hasattr(df_temp.columns, 'nlevels') and df_temp.columns.nlevels > 1:
                        df_temp = df_temp.xs(current_symbol, level='Ticker', axis=1, drop_level=True)

                    df_temp.reset_index(inplace=True)
                    column_mapping = {
                        "Datetime": "time", "Date": "time", "Open": "open", 
                        "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
                    }
                    df_temp.rename(columns=column_mapping, inplace=True)

                    # Calculate indicators
                    df_temp[f'SMA_{sma_short}'] = df_temp['close'].rolling(window=sma_short).mean()
                    df_temp[f'SMA_{sma_long}'] = df_temp['close'].rolling(window=sma_long).mean()
                    df_temp['SMA_20'] = df_temp['close'].rolling(window=20).mean()

                    # RSI calculation
                    delta = df_temp['close'].diff()
                    gain = np.where(delta > 0, delta, 0)
                    loss = np.where(delta < 0, -delta, 0)
                    avg_gain = pd.Series(gain, index=df_temp.index).rolling(window=14).mean()
                    avg_loss = pd.Series(loss, index=df_temp.index).rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    df_temp['RSI_14'] = 100 - (100 / (1 + rs))

                    # RSL calculation
                    df_temp['RSL_20'] = df_temp['close'] / df_temp['SMA_20']

                    # Signal generation
                    df_temp['Signal'] = 'Stay Out'
                    for i in range(1, len(df_temp)):
                        try:
                            rsi_up = df_temp['RSI_14'].iloc[i] > df_temp['RSI_14'].iloc[i-1]
                            rsi_down = df_temp['RSI_14'].iloc[i] < df_temp['RSI_14'].iloc[i-1]
                            rsl = df_temp['RSL_20'].iloc[i]
                            rsl_prev = df_temp['RSL_20'].iloc[i-1]

                            rsl_buy = (rsl > 1 and rsl > rsl_prev) or (rsl < 1 and rsl > rsl_prev)
                            rsl_sell = (rsl > 1 and rsl < rsl_prev) or (rsl < 1 and rsl < rsl_prev)

                            if (df_temp['close'].iloc[i] > df_temp[f'SMA_{sma_short}'].iloc[i] and 
                                df_temp['close'].iloc[i] > df_temp[f'SMA_{sma_long}'].iloc[i] and 
                                rsi_up and rsl_buy):
                                df_temp.at[i, 'Signal'] = 'Buy'
                            elif (df_temp['close'].iloc[i] < df_temp[f'SMA_{sma_short}'].iloc[i] and 
                                  rsi_down and rsl_sell):
                                df_temp.at[i, 'Signal'] = 'Sell'
                        except (IndexError, KeyError):
                            continue

                    current_signal = df_temp['Signal'].iloc[-1] if len(df_temp) > 0 else 'Stay Out'
                    previous_signal = df_temp['Signal'].iloc[-2] if len(df_temp) > 1 else 'Stay Out'
                    state_change = current_signal != previous_signal
                    current_price = df_temp['close'].iloc[-1] if len(df_temp) > 0 else 0

                    screening_results.append({
                        'symbol': current_symbol, 'status': 'OK',
                        'current_state': current_signal, 'previous_state': previous_signal,
                        'state_change': state_change, 'current_price': f"{current_price:.2f}"
                    })

                except Exception as e:
                    screening_results.append({
                        'symbol': current_symbol, 'status': f'Erro: {str(e)}',
                        'current_state': 'N/A', 'previous_state': 'N/A',
                        'state_change': False, 'current_price': 'N/A'
                    })

            progress_bar.progress(100)
            status_text.text("Análise de screening concluída!")

            # Display screening results
            st.header("📊 Resultados do Screening")
            results_df = pd.DataFrame(screening_results)
            state_changes = results_df[results_df['state_change'] == True]

            if not state_changes.empty:
                st.subheader("🔄 Ativos com Mudança de Estado")
                for _, row in state_changes.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    with col1:
                        st.write(f"**{row['symbol']}**")
                    with col2:
                        st.write(f"Anterior: {row['previous_state']}")
                    with col3:
                        st.write(f"Atual: {row['current_state']}")
                    with col4:
                        st.write(f"Preço: {row['current_price']}")

            with st.expander("📋 Todos os Resultados", expanded=False):
                st.dataframe(results_df, use_container_width=True)

        else:
            # Individual analysis mode
            status_text.text(f"Baixando dados para {symbol}...")
            
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            df = yf.download(symbol, start=start_str, end=end_str, interval=interval)

            if df is None or df.empty:
                st.error(f"Não foi possível obter dados para {symbol}. Verifique o ticker e tente novamente.")
                st.stop()

            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df = df.xs(symbol, level='Ticker', axis=1, drop_level=True)

            df.reset_index(inplace=True)
            column_mapping = {
                "Datetime": "time", "Date": "time", "Open": "open", 
                "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            }
            df.rename(columns=column_mapping, inplace=True)

            progress_bar.progress(25)
            status_text.text("Calculando indicadores técnicos...")

            # Calculate indicators
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

            progress_bar.progress(50)
            status_text.text("Gerando sinais de trading...")

            # Signal generation
            df['Signal'] = 'Stay Out'
            for i in range(1, len(df)):
                try:
                    rsi_up = df['RSI_14'].iloc[i] > df['RSI_14'].iloc[i-1]
                    rsi_down = df['RSI_14'].iloc[i] < df['RSI_14'].iloc[i-1]
                    rsl = df['RSL_20'].iloc[i]
                    rsl_prev = df['RSL_20'].iloc[i-1]

                    rsl_buy = (rsl > 1 and rsl > rsl_prev) or (rsl < 1 and rsl > rsl_prev)
                    rsl_sell = (rsl > 1 and rsl < rsl_prev) or (rsl < 1 and rsl < rsl_prev)

                    if (df['close'].iloc[i] > df[f'SMA_{sma_short}'].iloc[i] and 
                        df['close'].iloc[i] > df[f'SMA_{sma_long}'].iloc[i] and 
                        rsi_up and rsl_buy):
                        df.at[i, 'Signal'] = 'Buy'
                    elif (df['close'].iloc[i] < df[f'SMA_{sma_short}'].iloc[i] and 
                          rsi_down and rsl_sell):
                        df.at[i, 'Signal'] = 'Sell'
                except (IndexError, KeyError):
                    continue

            progress_bar.progress(75)
            status_text.text("Calculando retornos...")

            # Calculate returns
            returns_df = calculate_returns(df, exit_criteria, exit_params, trading_direction, exit_on_state_change)

            progress_bar.progress(100)
            status_text.text("Análise concluída!")

            # Display results
            st.header(f"📈 Análise Completa - {symbol}")

            # Create and display chart
            fig = create_chart(df, symbol, sma_short, sma_long)
            st.plotly_chart(fig, use_container_width=True)

            # Display returns analysis
            if not returns_df.empty:
                st.header("💰 Análise de Retornos")
                display_returns_section(returns_df, exit_criteria)
            else:
                st.warning("Nenhuma operação foi identificada no período analisado.")

            # Display signal summary
            st.header("📊 Resumo dos Sinais")
            buy_count = len(df[df['Signal'] == 'Buy'])
            sell_count = len(df[df['Signal'] == 'Sell'])
            stay_out_count = len(df[df['Signal'] == 'Stay Out'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sinais de Compra", buy_count)
            with col2:
                st.metric("Sinais de Venda", sell_count)
            with col3:
                st.metric("Fora do Mercado", stay_out_count)

    except Exception as e:
        st.error(f"Erro durante a análise: {str(e)}")
        import traceback
        st.error(f"Detalhes do erro: {traceback.format_exc()}")
    finally:
        progress_bar.progress(100)
        status_text.text("Análise finalizada!")

else:
    st.info("👆 Configure os parâmetros na barra lateral e clique em 'Analisar' para começar.")
