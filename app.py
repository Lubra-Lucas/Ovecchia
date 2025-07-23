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
                exit_date = row['exit_time'].strftime('%d/%m/%Y %H:%M') if hasattr(row['exit_time'], 'strftime') else str(exit_time)
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
    page_title="OVECCHIA TRADING - MODELO QUANT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-buy {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    .status-sell {
        background: linear-gradient(90deg, #f44336, #da190b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    .status-out {
        background: linear-gradient(90deg, #9E9E9E, #757575);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Sidebar improvements */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown('<h1 class="main-title">📈 OVECCHIA TRADING - MODELO QUANT</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Sistema Avançado de Análise Técnica e Sinais de Trading</p>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("### 🎯 Parâmetros de Análise")
st.sidebar.markdown("---")

# Input fields in sidebar
with st.sidebar:
    st.markdown("#### 🔍 Modo de Análise")
    analysis_mode = st.radio(
        "Escolha o tipo de análise:",
        ["Ativo Individual", "Screening de Múltiplos Ativos"]
    )
    st.markdown("---")

    if analysis_mode == "Ativo Individual":
        st.markdown("#### 💹 Configuração de Ativo")
        # Symbol input with examples
        symbol = st.text_input(
            "Ticker",
            value="BTC-USD",
            help="Examples: BTC-USD, PETR4.SA, AAPL, EURUSD=X"
        ).strip()

    else:  # Screening mode
        st.markdown("#### 📊 Lista de Ativos para Screening")

        # Predefined lists
        preset_lists = {
            "Criptomoedas": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD",
                                   "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD",
                                   "TRX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
                                   "FIL-USD", "APT-USD", "ARB-USD", "NEAR-USD", "VET-USD"],
            "Ações Brasileiras": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "B3SA3.SA",
                                 "ABEV3.SA", "BBAS3.SA", "WEGE3.SA", "PETR3.SA", "JBSS3.SA",
                                 "ITSA4.SA", "ELET3.SA", "SUZB3.SA", "GGBR4.SA", "RENT3.SA",
                                 "VBBR3.SA", "BRFS3.SA", "RAIL3.SA", "ELET6.SA", "CCRO3.SA"],
            "Ações Americanas": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "BABA"],
            "Pares de Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X"],
            "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "KE=F", "CC=F", "KC=F"]
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

        # Show selected symbols
        st.write(f"**{len(symbols_list)} ativos selecionados para screening**")

    # Date range selection
    st.markdown("---")
    st.markdown("#### 📅 Intervalo de Data")

    # Default date range (last 30 days)
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
    st.markdown("#### ⏱️ Intervalo de Tempo (Timeframe)")
    interval_options = {
        "1 minute": "1m",
        "2 minutes": "2m",
        "5 minutes": "5m",
        "15 minutes": "15m",
        "30 minutes": "30m",
        "60 minutes": "60m",
        "90 minutes": "90m",
        "4 hours": "4h",
        "1 day": "1d",
        "5 days": "5d",
        "1 week": "1wk",
        "1 month": "1mo",
        "3 months": "3mo"
    }

    interval_display = st.selectbox(
        "Selecione o Intervalo",
        list(interval_options.keys()),
        index=9  # Default to "1 day"
    )
    interval = interval_options[interval_display]

    # Data limitations info
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h"]:
        st.info(
            "ℹ️ **Limitação de Dados:** Dados intradiários (intervalos de minuto/hora) "
            "são limitados aos últimos 60 a 730 dias, dependendo do intervalo. "
            "Para análises históricas além desse período, use intervalos diários."
        )

    # Confirmation candles parameter
    st.markdown("---")
    st.markdown("#### ✅ Confirmação de Sinais")
    confirm_candles = st.number_input(
        "Candles de Confirmação",
        min_value=0,
        max_value=5,
        value=0,
        help="Número de candles consecutivos para confirmar o sinal"
    )

    # Moving averages configuration
    st.markdown("#### 📈 Configuração de Médias Móveis")
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
            min_value=9,
            max_value=300,
            value=70,
            step=5,
            help="Segunda condição para sinais de compra"
        )

    if st.session_state.get('is_selecting_moving_averages', False):
        # Display a notice about moving averages and signal generation
        st.info(
            "ℹ️ **Nota:** As médias móveis influenciam na quantidade e qualidade dos sinais. "
            "Se usar médias longas, atente-se ao intervalo de data para garantir a geração de sinais de forma eficaz.")
    else:
        st.session_state['is_selecting_moving_averages'] = True


    # Trading direction configuration
    st.markdown("---")
    st.markdown("#### 🎯 Direção de Operação")

    trading_direction = st.selectbox(
        "Escolha a direção das operações:",
        ["Ambos (Compra e Venda)", "Apenas Comprado", "Apenas Vendido"],
        index=0,
        help="Selecione se deseja operar apenas comprado, apenas vendido, ou ambas direções"
    )

    # Exit criteria configuration
    st.markdown("#### 🚪 Critérios de Saída Personalizados")

    exit_criteria = st.selectbox(
        "Tipo de Saída",
        ["Mudança de Estado", "Stop Loss", "Alvo Fixo", "Tempo", "Média Móvel"],
        index=0,
        help="Escolha como deseja sair das posições"
    )

    # Add a checkbox to decide whether to include state change in exit criteria
    include_state_change = st.checkbox(
        "Sair por mudança de estado?",
        value=True,
        help="Selecione se a operação deve ser encerrada quando houver mudança de estado, além do critério de saída."
    )

    # Optimization option
    optimize_params = st.checkbox(
        "🎯 Otimizar Parâmetros",
        value=False,
        help="Testa diferentes combinações de parâmetros para encontrar o melhor retorno"
    )

    # Additional parameters based on exit criteria
    exit_params = {}

    if exit_criteria == "Stop Loss":
        if not optimize_params:
            exit_params['stop_type'] = st.selectbox(
                "Tipo de Stop",
                ["Stop Justo", "Stop Balanceado", "Stop Largo"]
            )
        else:
            st.info("🔍 Modo Otimização: Testará todos os tipos de stop (Justo, Balanceado, Largo)")
    elif exit_criteria == "Alvo Fixo":
        if not optimize_params:
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
        else:
            st.info("🔍 Modo Otimização: Testará múltiplas combinações de alvo e stop")
            col1, col2 = st.columns(2)
            with col1:
                target_range = [float(t.strip()) for t in st.text_input(
                    "Alvos a Testar (%)",
                    value="2.0;3.0;4.0;5.0",
                    help="Digite os alvos separados por ponto e vírgula, ex: 1.0;2.5;5.0"
                ).split(';') if t.strip()]

            with col2:
                stop_range = [float(s.strip()) for s in st.text_input(
                    "Stops a Testar (%)",
                    value="1.0;2.0;3.0",
                    help="Digite os stops separados por ponto e vírgula, ex: 0.5;1.0;3.0"
                ).split(';') if s.strip()]
            exit_params['target_range'] = target_range
            exit_params['stop_range'] = stop_range
    elif exit_criteria == "Tempo":
        if not optimize_params:
            exit_params['time_candles'] = st.number_input(
                "Candles após entrada",
                min_value=1,
                max_value=1000,
                value=10,
                step=1,
                help="Número de candles após a entrada para sair da posição"
            )
        else:
            st.info("🔍 Modo Otimização: Testará de 1 a X candles que usuário definir")
            max_candles = st.number_input(
                "Máximo de candles a testar",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )
            exit_params['max_candles'] = max_candles
    elif exit_criteria == "Média Móvel":
        if not optimize_params:
            exit_params['ma_period'] = st.number_input(
                "Período da Média Móvel",
                min_value=5,
                max_value=200,
                value=20,
                step=5,
                help="Período para a média móvel (MM)"
            )
        else:
            st.info("🔍 Modo Otimização: Testará diferentes períodos de MM")
            ma_input = st.text_input(
                "Digite os períodos de MM separados por ponto e vírgula:",
                value="10;20;50",
                help="Exemplo: 10;20;50"
            )
            ma_range = [int(x.strip()) for x in ma_input.split(';') if x.strip()]
            exit_params['ma_range'] = ma_range

    # Analyze button
    st.markdown("---")
    analyze_button = st.button("🚀 INICIAR ANÁLISE", type="primary", use_container_width=True)

# Main content area
if analyze_button:
    if analysis_mode == "Ativo Individual":
        if not symbol:
            st.error("Por favor entre com um ticker válido.")
            st.stop()
        symbols_to_analyze = [symbol]
    else:  # Screening mode
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
                    # Convert dates to strings
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")

                    # Download data for current symbol
                    df_temp = yf.download(current_symbol, start=start_str, end=end_str, interval=interval)

                    if df_temp is None or df_temp.empty:
                        screening_results.append({
                            'symbol': current_symbol,
                            'status': 'Erro - Sem dados',
                            'current_state': 'N/A',
                            'previous_state': 'N/A',
                            'state_change': False,
                            'current_price': 'N/A'
                        })
                        continue

                    # Handle multi-level columns if present
                    if hasattr(df_temp.columns, 'nlevels') and df_temp.columns.nlevels > 1:
                        df_temp = df_temp.xs(current_symbol, level='Ticker', axis=1, drop_level=True)

                    # Ensure we have the required columns
                    df_temp.reset_index(inplace=True)
                    column_mapping = {
                        "Datetime": "time", 
                        "Date": "time", 
                        "Open": "open", 
                        "High": "high", 
                        "Low": "low", 
                        "Close": "close",
                        "Volume": "volume"
                    }
                    df_temp.rename(columns=column_mapping, inplace=True)

                    # Calculate indicators (simplified for screening)
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
                        rsi_up = df_temp['RSI_14'].iloc[i] > df_temp['RSI_14'].iloc[i-1]
                        rsi_down = df_temp['RSI_14'].iloc[i] < df_temp['RSI_14'].iloc[i-1]
                        rsl = df_temp['RSL_20'].iloc[i]
                        rsl_prev = df_temp['RSL_20'].iloc[i-1]

                        rsl_buy = (rsl > 1 and rsl > rsl_prev) or (rsl < 1 and rsl > rsl_prev)
                        rsl_sell = (rsl > 1 and rsl < rsl_prev) or (rsl < 1 and rsl < rsl_prev)

                        if (
                            df_temp['close'].iloc[i] > df_temp[f'SMA_{sma_short}'].iloc[i]
                            and df_temp['close'].iloc[i] > df_temp[f'SMA_{sma_long}'].iloc[i]
                            and rsi_up and rsl_buy
                        ):
                            df_temp.at[i, 'Signal'] = 'Buy'
                        elif (
                            df_temp['close'].iloc[i] < df_temp[f'SMA_{sma_short}'].iloc[i]
                            and rsi_down and rsl_sell
                        ):
                            df_temp.at[i, 'Signal'] = 'Sell'

                    # State persistence with confirmation delay
                    df_temp['Estado'] = 'Stay Out'

                    for i in range(len(df_temp)):
                        if i == 0:
                            # Primeiro candle sempre Stay Out
                            continue

                        # Estado anterior
                        estado_anterior = df_temp['Estado'].iloc[i - 1]

                        # Verificar se houve mudança de sinal há confirm_candles períodos atrás
                        if confirm_candles == 0:
                            # Sem confirmação - aplicar sinal imediatamente
                            sinal_atual = df_temp['Signal'].iloc[i]
                            if sinal_atual != 'Stay Out':
                                df_temp.loc[df_temp.index[i], 'Estado'] = sinal_atual
                            else:
                                df_temp.loc[df_temp.index[i], 'Estado'] = estado_anterior
                        else:
                            # Com confirmação - aplicar sinal de confirm_candles períodos atrás
                            if i > confirm_candles:
                                # Verificar o sinal de confirm_candles períodos atrás
                                sinal_passado = df_temp['Signal'].iloc[i - confirm_candles]
                                if sinal_passado != 'Stay Out':
                                    df_temp.loc[df_temp.index[i], 'Estado'] = sinal_passado
                                else:
                                    df_temp.loc[df_temp.index[i], 'Estado'] = estado_anterior
                            else:
                                # Ainda não temos candles suficientes, manter estado anterior
                                df_temp.loc[df_temp.index[i], 'Estado'] = estado_anterior

                    # Check for state change
                    current_state = df_temp['Estado'].iloc[-1]
                    previous_state = df_temp['Estado'].iloc[-2] if len(df_temp) > 1 else current_state
                    state_change = current_state != previous_state
                    current_price = df_temp['close'].iloc[-1]

                    screening_results.append({
                        'symbol': current_symbol,
                        'status': 'Sucesso',
                        'current_state': current_state,
                        'previous_state': previous_state,
                        'state_change': state_change,
                        'current_price': current_price
                    })

                except Exception as e:
                    screening_results.append({
                        'symbol': current_symbol,
                        'status': f'Erro: {str(e)[:50]}...',
                        'current_state': 'N/A',
                        'previous_state': 'N/A',
                        'state_change': False,
                        'current_price': 'N/A'
                    })

            progress_bar.progress(100)
            status_text.text("Screening Completo!")

            # Display screening results
            st.success(f"✅ Screening completo para {len(symbols_to_analyze)} ativos")

            # Filter and display assets with state changes
            state_changes = [r for r in screening_results if r['state_change']]

            if state_changes:
                st.subheader(f"🚨 {len(state_changes)} Ativo(s) com Mudança de Estado Detectada!")

                for result in state_changes:
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

                    state_icon = "🔵" if result['current_state'] == "Buy" else "🔴" if result['current_state'] == "Sell" else "⚫"
                    prev_icon = "🔵" if result['previous_state'] == "Buy" else "🔴" if result['previous_state'] == "Sell" else "⚫"

                    with col1:
                        st.write(f"**{result['symbol']}**")
                    with col2:
                        st.write(f"Preço: {result['current_price']:.2f}")
                    with col3:
                        st.write(f"De: {prev_icon} {result['previous_state']}")
                    with col4:
                        st.write(f"Para: {state_icon} {result['current_state']}")
                    with col5:
                        if result['current_state'] == 'Buy':
                            st.success("🟢 COMPRA")
                        elif result['current_state'] == 'Sell':
                            st.error("🔴 VENDA")
                        else:
                            st.info("⚫ FORA")

                    st.markdown("---")
            else:
                st.info("ℹ️ Nenhum ativo com mudança de estado detectada no período analisado.")

            # Summary table of all assets
            st.subheader("📊 Resumo Geral do Screening")

            # Create summary dataframe
            summary_df = pd.DataFrame(screening_results)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_assets = len(summary_df)
                st.metric("Total de Ativos", total_assets)

            with col2:
                successful_analysis = len(summary_df[summary_df['status'] == 'Sucesso'])
                st.metric("Análises Bem-sucedidas", successful_analysis)

            with col3:
                buy_signals = len(summary_df[summary_df['current_state'] == 'Buy'])
                st.metric("Sinais de Compra", buy_signals)

            with col4:
                sell_signals = len(summary_df[summary_df['current_state'] == 'Sell'])
                st.metric("Sinais de Venda", sell_signals)

            # Display full table
            st.dataframe(summary_df, use_container_width=True)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

        else:
            # Individual analysis mode (existing code)
            # Fetch data
            status_text.text("Coletando dados de mercado...")
            progress_bar.progress(20)

            # Convert dates to strings
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Download data
            df = yf.download(symbol, start=start_str, end=end_str, interval=interval)

            if df is None or df.empty:
                st.error(f"Sem data encontrada para '{symbol}' nesse período de tempo.")
                st.stop()

            # Handle multi-level columns if present
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df = df.xs(symbol, level='Ticker', axis=1, drop_level=True)

            progress_bar.progress(40)
            status_text.text("Processando indicadores...")

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
            status_text.text("Gerando sinais de trading...")

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

            # State persistence with confirmation delay
            df['Estado'] = 'Stay Out'

            for i in range(len(df)):
                if i == 0:
                    # Primeiro candle sempre Stay Out
                    continue

                # Estado anterior
                estado_anterior = df['Estado'].iloc[i - 1]

                # Verificar se houve mudança de sinal há confirm_candles períodos atrás
                if confirm_candles == 0:
                    # Sem confirmação - aplicar sinal imediatamente
                    sinal_atual = df['Signal'].iloc[i]
                    if sinal_atual != 'Stay Out':
                        df.loc[df.index[i], 'Estado'] = sinal_atual
                    else:
                        df.loc[df.index[i], 'Estado'] = estado_anterior
                else:
                    # Com confirmação - aplicar sinal de confirm_candles períodos atrás
                    if i > confirm_candles:
                        # Verificar o sinal de confirm_candles períodos atrás
                        sinal_passado = df['Signal'].iloc[i - confirm_candles]
                        if sinal_passado != 'Stay Out':
                            df.loc[df.index[i], 'Estado'] = sinal_passado
                        else:
                            df.loc[df.index[i], 'Estado'] = estado_anterior
                    else:
                        # Ainda não temos candles suficientes, manter estado anterior
                        df.loc[df.index[i], 'Estado'] = estado_anterior

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
            fatores = {'Stop_Justo': 2.0 , 'Stop_Balanceado': 2.5 , 'Stop_Largo': 3.5}

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
            def calculate_signal_returns(df, direction="Ambos (Compra e Venda)"):
                returns_data = []
                current_signal = None
                entry_price = None
                entry_time = None

                for i in range(len(df)):
                    estado = df['Estado'].iloc[i]
                    price = df['close'].iloc[i]
                    time = df['time'].iloc[i]

                    # Filter signals based on trading direction
                    should_enter = False
                    if direction == "Ambos (Compra e Venda)":
                        should_enter = estado in ['Buy', 'Sell']
                    elif direction == "Apenas Comprado":
                        should_enter = estado == 'Buy'
                    elif direction == "Apenas Vendido":
                        should_enter = estado == 'Sell'

                    if estado != current_signal and should_enter:
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
                    elif (estado == 'Stay Out' or 
                          (direction == "Apenas Comprado" and estado == 'Sell') or
                          (direction == "Apenas Vendido" and estado == 'Buy')) and current_signal is not None:
                        # Exit position to stay out or opposite signal
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

            returns_df = calculate_signal_returns(df, trading_direction)

            # Calculate custom exit criteria returns
            def calculate_custom_exit_returns(df, exit_criteria, exit_params, direction="Ambos (Compra e Venda)", include_state_change=True):
                if exit_criteria == "Mudança de Estado":
                    return calculate_signal_returns(df, direction)

                custom_returns = []
                current_signal = None
                entry_price = None
                entry_time = None
                entry_index = None
                previous_state = None  # Track previous state to detect changes

                for i in range(len(df)):
                    estado = df['Estado'].iloc[i]
                    price = df['close'].iloc[i]
                    time = df['time'].iloc[i]

                    # Filter signals based on trading direction
                    should_enter = False
                    should_exit_on_opposite = False

                    if direction == "Ambos (Compra e Venda)":
                        should_enter = estado in ['Buy', 'Sell']
                    elif direction == "Apenas Comprado":
                        should_enter = estado == 'Buy'
                        should_exit_on_opposite = estado == 'Sell'
                    elif direction == "Apenas Vendido":
                        should_enter = estado == 'Sell'
                        should_exit_on_opposite = estado == 'Buy'

                    # Check if we have an active position
                    if current_signal is not None and entry_price is not None and entry_index is not None:

                        # 1. Check for exit conditions based on include_state_change setting
                        should_exit_by_state = False
                        exit_reason_state = None
                        
                        if include_state_change:
                            if direction == "Ambos (Compra e Venda)":
                                # For "Ambos", exit on state change to Stay Out OR opposite signal
                                if estado == 'Stay Out':
                                    should_exit_by_state = True
                                    exit_reason_state = 'Mudança de Estado'
                                elif estado != current_signal and estado in ['Buy', 'Sell']:
                                    should_exit_by_state = True
                                    exit_reason_state = 'Mudança de Estado'
                            else:
                                # For single direction, exit on Stay Out or opposite signal
                                if estado == 'Stay Out' or should_exit_on_opposite:
                                    should_exit_by_state = True
                                    exit_reason_state = 'Mudança de Estado' if estado == 'Stay Out' else 'Mudança de Estado'

                        if should_exit_by_state:
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
                                'exit_reason': exit_reason_state
                            })

                            # Start new position if criteria met and direction allows
                            if should_enter and previous_state != estado:
                                current_signal = estado
                                entry_price = price
                                entry_time = time
                                entry_index = i
                            else:
                                current_signal = None
                                entry_price = None
                                entry_time = None
                                entry_index = None

                            previous_state = estado
                            continue

                        # 2. Check custom exit criteria (only if no state change exit occurred)
                        exit_price, exit_time_custom, exit_reason = calculate_exit(
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
                                'exit_time': exit_time_custom,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'return_pct': return_pct,
                                'exit_reason': exit_reason
                            })

                            # Position closed by custom criteria - wait for next state change
                            current_signal = None
                            entry_price = None
                            entry_time = None
                            entry_index = None
                            continue

                    # Entry logic - ONLY open new position on STATE CHANGE to allowed signals
                    elif previous_state != estado and should_enter:
                        # This is a state change to allowed signal - open new position
                        current_signal = estado
                        entry_price = price
                        entry_time = time
                        entry_index = i

                    # Update previous state for next iteration
                    previous_state = estado

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
                elif criteria == "Média Móvel":
                    ma_period = params['ma_period']
                    ma = df['close'].rolling(window=ma_period).mean()
                    for i in range(entry_idx + 1, min(current_idx + 1, len(df))):
                        current_price = df['close'].iloc[i]
                        ma_value = ma.iloc[i]

                        if signal == 'Buy' and current_price < ma_value:
                            return current_price, df['time'].iloc[i], f"MM{ma_period} Cruzada para Baixo"
                        elif signal == 'Sell' and current_price > ma_value:
                            return current_price, df['time'].iloc[i], f"MM{ma_period} Cruzada para Cima"

                return None, None, None

            def optimize_exit_parameters(df, criteria, params, direction="Ambos (Compra e Venda)"):
                """Optimize parameters for the selected exit criteria"""
                all_results = []
                best_return = float('-inf')
                best_params = None
                best_returns_df = pd.DataFrame()

                include_state_change_options = [True, False] if trading_direction == "Ambos (Compra e Venda)" else [include_state_change]

                if criteria == "Tempo":
                    # Test different number of candles
                    max_candles = params.get('max_candles', 20)
                    for candles in range(1, max_candles + 1):
                        test_params = {'time_candles': candles}
                        returns_df = calculate_custom_exit_returns(df, criteria, test_params, direction, include_state_change)

                        if not returns_df.empty:
                            total_return = returns_df['return_pct'].sum()
                            avg_return = returns_df['return_pct'].mean()
                            win_rate = (returns_df['return_pct'] > 0).sum() / len(returns_df) * 100

                            all_results.append({
                                'parametro': f"{candles} candles",
                                'total_return': total_return,
                                'avg_return': avg_return,
                                'win_rate': win_rate,
                                'total_trades': len(returns_df)
                            })

                            if total_return > best_return:
                                best_return = total_return
                                best_params = candles
                                best_returns_df = returns_df.copy()

                elif criteria == "Média Móvel":
                    # Test different MA periods
                    ma_range = params.get('ma_range', [10, 20, 50])
                    for ma_period in ma_range:
                        test_params = {'ma_period': ma_period}
                        returns_df = calculate_custom_exit_returns(df, criteria, test_params, direction, include_state_change)

                        if not returns_df.empty:
                            total_return = returns_df['return_pct'].sum()
                            avg_return = returns_df['return_pct'].mean()
                            win_rate = (returns_df['return_pct'] > 0).sum() / len(returns_df) * 100

                            all_results.append({
                                'parametro': f"MM{ma_period}",
                                'total_return': total_return,
                                'avg_return': avg_return,
                                'win_rate': win_rate,
                                'total_trades': len(returns_df)
                            })

                            if total_return > best_return:
                                best_return = total_return
                                best_params = ma_period
                                best_returns_df = returns_df.copy()

                elif criteria == "Stop Loss":
                    # Test different stop types
                    stop_types = ["Stop Justo", "Stop Balanceado", "Stop Largo"]
                    for stop_type in stop_types:
                        test_params = {'stop_type': stop_type}
                        returns_df = calculate_custom_exit_returns(df, criteria, test_params, direction, include_state_change)

                        if not returns_df.empty:
                            total_return = returns_df['return_pct'].sum()
                            avg_return = returns_df['return_pct'].mean()
                            win_rate = (returns_df['return_pct'] > 0).sum() / len(returns_df) * 100

                            all_results.append({
                                'parametro': stop_type,
                                'total_return': total_return,
                                'avg_return': avg_return,
                                'win_rate': win_rate,
                                'total_trades': len(returns_df)
                            })

                            if total_return > best_return:
                                best_return = total_return
                                best_params = stop_type
                                best_returns_df = returns_df.copy()

                elif criteria == "Alvo Fixo":
                    # Test different combinations of target and stop
                    target_range = params.get('target_range', [2.0, 3.0, 4.0, 5.0])
                    stop_range = params.get('stop_range', [1.0, 2.0, 3.0])

                    for target in target_range:
                        for stop in stop_range:
                            if target > stop:  # Only test valid combinations
                                test_params = {'target_pct': target, 'stop_loss_pct': stop}
                                returns_df = calculate_custom_exit_returns(df, criteria, test_params, direction, include_state_change)

                                if not returns_df.empty:
                                    total_return = returns_df['return_pct'].sum()
                                    avg_return = returns_df['return_pct'].mean()
                                    win_rate = (returns_df['return_pct'] > 0).sum() / len(returns_df) * 100

                                    all_results.append({
                                        'parametro': f"Stop {stop}% / Alvo {target}%",
                                        'total_return': total_return,
                                        'avg_return': avg_return,
                                        'win_rate': win_rate,
                                        'total_trades': len(returns_df)
                                    })

                                    if total_return > best_return:
                                        best_return = total_return
                                        best_params = {'stop': stop, 'target': target}
                                        best_returns_df = returns_df.copy()

                return {
                    'best_returns': best_returns_df,
                    'best_params': best_params,
                    'best_total_return': best_return,
                    'all_results': all_results
                }

            # Calculate returns with optimization if enabled
            if optimize_params:
                status_text.text("Otimizando parâmetros...")
                progress_bar.progress(85)

                optimization_results = optimize_exit_parameters(df, exit_criteria, exit_params, trading_direction)
                custom_returns_df = optimization_results['best_returns']
                best_params = optimization_results['best_params']
                all_results = optimization_results['all_results']
            else:
                custom_returns_df = calculate_custom_exit_returns(df, exit_criteria, exit_params, trading_direction, include_state_change)
                optimization_results = None

            progress_bar.progress(100)
            status_text.text("Análise Completa!")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Display results
            if optimize_params and optimization_results:
                st.success(f"✅ Análise e otimização completa para {symbol_label}")

                # Show optimization results
                st.subheader("🎯 Resultados da Otimização")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Melhor Retorno Total", f"{optimization_results['best_total_return']:.2f}%")
                with col2:
                    if exit_criteria == "Tempo":
                        st.metric("Melhor Parâmetro", f"{best_params} candles")
                    elif exit_criteria == "Stop Loss":
                        st.metric("Melhor Stop", best_params)
                    elif exit_criteria == "Alvo Fixo":
                        st.metric("Melhor Combinação", f"Stop {best_params['stop']}% / Alvo {best_params['target']}%")
                    elif exit_criteria == "Média Móvel":
                        st.metric("Melhor Período MM", f"MM{best_params}")
                with col3:
                    st.metric("Operações", len(custom_returns_df))

                # Show comparison table
                st.subheader("📊 Comparação de Parâmetros")
                comparison_df = pd.DataFrame(all_results)
                comparison_df = comparison_df.sort_values('total_return', ascending=False)
                st.dataframe(comparison_df, use_container_width=True)

            else:
                st.success(f"✅ Análise completa para  {symbol_label}")

            # Current status display with improved styling
            st.markdown("### 📊 Status Atual do Mercado")
            
            col1, col2, col3, col4 = st.columns(4)

            current_price = df['close'].iloc[-1]
            current_signal = df['Estado'].iloc[-1]
            current_rsi = df['RSI_14'].iloc[-1]
            current_rsl = df['RSL_20'].iloc[-1]

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #1f77b4;">💰 Preço Atual</h4>
                    <h2 style="margin: 0; color: #333;">{current_price:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                signal_class = "status-buy" if current_signal == "Buy" else "status-sell" if current_signal == "Sell" else "status-out"
                signal_icon = "🔵" if current_signal == "Buy" else "🔴" if current_signal == "Sell" else "⚫"
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #1f77b4;">🎯 Sinal Atual</h4>
                    <div class="{signal_class}">{signal_icon} {current_signal}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                rsi_color = "#4CAF50" if current_rsi > 50 else "#f44336"
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #1f77b4;">📈 RSI (14)</h4>
                    <h2 style="margin: 0; color: {rsi_color};">{current_rsi:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                rsl_color = "#4CAF50" if current_rsl > 1 else "#f44336"
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #1f77b4;">📊 RSL (20)</h4>
                    <h2 style="margin: 0; color: {rsl_color};">{current_rsl:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Create the interactive chart
            titulo_grafico = f"LUBRA TRADING - {symbol_label} - Timeframe: {interval.upper()}"

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
                subplot_titles=("Gráfico do Preço com Sinais", "Indicador de Sinais")
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
                line=dict(color='rgba(0,0,0,0)'),name='Price',
                hovertemplate="<b>Price:</b> %{y:.2f}<br><b>Time:</b> %{x}<extra></extra>",
                showlegend=False
            ), row=1, col=1)

            # Add all stop loss traces
            stop_colors = {
                "Stop_Justo": "orange",
                "Stop_Balanceado": "gray",
                "Stop_Largo": "green"
            }

            for stop_type, color in stop_colors.items():
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df[stop_type],
                    mode="lines", name=stop_type.replace("_", " "),
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=f"<b>{stop_type.replace('_', ' ')}:</b> %{{y:.2f}}<extra></extra>"
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
                name='Sinal de Compra'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color='red', width=2),
                name='Sinal de Venda'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color='black', width=2),
                name='Ficar de Fora'
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
                            ticktext=['Venda', 'Ficar de Fora', 'Compra'], row=2, col=1)
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
            st.markdown("---")
            st.markdown("## 📈 Análise de Retornos")
            st.markdown("Análise detalhada dos retornos baseada nos diferentes critérios de saída")

            # Create tabs for different return calculations
            direction_label = trading_direction.replace("Ambos (Compra e Venda)", "Ambos").replace("Apenas ", "")

            if optimize_params and optimization_results:
                tab1, tab2, tab3 = st.tabs([f"📊 Mudança de Estado - {direction_label}", f"🎯 {exit_criteria} (Otimizado) - {direction_label}", "📋 Comparação Detalhada"])
            else:
                tab1, tab2 = st.tabs([f"📊 Mudança de Estado - {direction_label}", f"🎯 {exit_criteria} - {direction_label}"])

            with tab1:
                st.write(f"**Retornos baseados na mudança natural do estado dos sinais - {trading_direction}**")
                if not returns_df.empty:
                    display_returns_section(returns_df, "Mudança de Estado")
                else:
                    st.info(f"Nenhuma operação completa encontrada no período analisado para a direção: {trading_direction}.")

            with tab2:
                if optimize_params and optimization_results:
                    st.write(f"**Retornos otimizados para: {exit_criteria} - {trading_direction}**")
                    if best_params:
                        if exit_criteria == "Tempo":
                            st.success(f"🏆 Melhor configuração: **{best_params} candles**")
                        elif exit_criteria == "Stop Loss":
                            st.success(f"🏆 Melhor configuração: **{best_params}**")
                        elif exit_criteria == "Alvo Fixo":
                            st.success(f"🏆 Melhor configuração: **Stop {best_params['stop']}% / Alvo {best_params['target']}%**")
                        elif exit_criteria == "Média Móvel":
                            st.success(f"🏆 Melhor configuração: **MM{best_params}**")
                else:
                    st.write(f"**Retornos baseados no critério: {exit_criteria} - {trading_direction}**")

                if not custom_returns_df.empty:
                    display_returns_section(custom_returns_df, exit_criteria)
                else:
                    st.info(f"Nenhuma operação completa encontrada com este critério no período analisado para a direção: {trading_direction}.")

            if optimize_params and optimization_results:
                with tab3:
                    st.write("**Comparação detalhada de todos os parâmetros testados**")

                    # Create a more detailed comparison
                    if all_results:
                        comparison_df = pd.DataFrame(all_results)
                        comparison_df = comparison_df.sort_values('total_return', ascending=False)

                        # Format columns
                        comparison_df['total_return'] = comparison_df['total_return'].round(2)
                        comparison_df['avg_return'] = comparison_df['avg_return'].round(2)
                        comparison_df['win_rate'] = comparison_df['win_rate'].round(1)

                        # Rename columns for better display
                        comparison_df.columns = ['Parâmetro', 'Retorno Total (%)', 'Retorno Médio (%)', 'Taxa de Acerto (%)', 'Total de Operações']

                        # Color code the best result
                        def highlight_best(s):
                            if s.name == 'Retorno Total (%)':
                                is_max = s == s.max()
                                return ['background-color: lightgreen' if v else '' for v in is_max]
                            return ['' for _ in s]

                        styled_df = comparison_df.style.apply(highlight_best, axis=0)
                        st.dataframe(styled_df, use_container_width=True)

                        # Show summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Melhor Retorno Total", f"{comparison_df['Retorno Total (%)'].max():.2f}%")
                        with col2:
                            st.metric("Pior Retorno Total", f"{comparison_df['Retorno Total (%)'].min():.2f}%")
                        with col3:
                            st.metric("Diferença", f"{comparison_df['Retorno Total (%)'].max() - comparison_df['Retorno Total (%)'].min():.2f}%")
                    else:
                        st.info("Nenhum resultado de otimização disponível.")

            st.markdown("---")
            # Technical analysis summary with improved layout
            st.markdown("## 📋 Informações Técnicas")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🛡️ Níveis de Stop Loss")
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>🔴 Stop Justo:</strong> {df['Stop_Justo'].iloc[-1]:.2f}</p>
                    <p><strong>🟡 Stop Balanceado:</strong> {df['Stop_Balanceado'].iloc[-1]:.2f}</p>
                    <p><strong>🟢 Stop Largo:</strong> {df['Stop_Largo'].iloc[-1]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                buy_signals = (df['Estado'] == 'Buy').sum()
                sell_signals = (df['Estado'] == 'Sell').sum()
                stay_out = (df['Estado'] == 'Stay Out').sum()
                
                st.markdown("### 📊 Distribuição dos Sinais")
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>🔵 Sinais de Compra:</strong> {buy_signals}</p>
                    <p><strong>🔴 Sinais de Venda:</strong> {sell_signals}</p>
                    <p><strong>⚫ Fora do Mercado:</strong> {stay_out}</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.write("Please check your inputs and try again.")

else:
    # Initial state - show instructions with better styling
    st.markdown("""
    <div style="background: linear-gradient(90deg, #e3f2fd, #f3e5f5); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #1976d2; margin-bottom: 1rem;">🚀 Bem-vindo ao Sistema de Trading Quant!</h2>
        <p style="font-size: 1.2rem; color: #666;">Configure os parâmetros na barra lateral e inicie sua análise profissional</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Como Usar")
        st.markdown("""
        <div class="metric-card">
            <ol>
                <li><strong>🎯 Insira o Símbolo do Ativo</strong><br>Use símbolos padrão (ex: BTC-USD, AAPL, PETR4.SA)</li><br>
                <li><strong>📅 Defina o Período de Análise</strong><br>Escolha o intervalo de datas desejado</li><br>
                <li><strong>⏱️ Selecione o Intervalo de Tempo</strong><br>Escolha o timeframe para análise</li><br>
                <li><strong>✅ Configure a Confirmação</strong><br>Defina quantos sinais consecutivos necessários</li><br>
                <li><strong>🚀 Clique em Analisar</strong><br>Gere sua análise profissional</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🔍 Funcionalidades")
        st.markdown("""
        <div class="metric-card">
            <ul>
                <li><strong>📈 Indicadores Técnicos</strong><br>SMA (20, 60, 70), RSI (14), RSL (20), ATR (14)</li><br>
                <li><strong>🎯 Sinais de Trading</strong><br>Compra, Venda e Ficar de Fora com confirmação</li><br>
                <li><strong>🛡️ Níveis de Stop Loss</strong><br>Três níveis baseados no ATR</li><br>
                <li><strong>📊 Gráficos Interativos</strong><br>Zoom, arrastar e hover para detalhes</li><br>
                <li><strong>📡 Dados em Tempo Real</strong><br>API do Yahoo Finance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; margin-top: 2rem;">
    <p style="color: #666; margin: 0;"><strong>OVECCHIA TRADING - MODELO QUANT</strong></p>
    <p style="color: #999; font-size: 0.9rem; margin: 0;">⚠️ Para fins educacionais apenas. Não constitui recomendação financeira.</p>
</div>
""", unsafe_allow_html=True)