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

# Main title
st.title("OVECCHIA TRADING - MODELO QUANT ")
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
        # Symbol input with examples
        symbol = st.text_input(
            "Ticker",
            value="BTC-USD",
            help="Examples: BTC-USD, PETR4.SA, AAPL, EURUSD=X"
        ).strip()

    else:  # Screening mode
        st.subheader("Lista de Ativos para Screening")

        # Predefined lists
        preset_lists = {
            "Criptomoedas Top 10": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD",
                                   "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD",
                                   "TRX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
                                   "ICP-USD", "TON11419-USD", "ATOM-USD", "ETC-USD", "XLM-USD",
                                   "FIL-USD", "HBAR-USD", "APT-USD", "ARB-USD", "INJ-USD",
                                   "NEAR-USD", "VET-USD", "RNDR-USD", "OP-USD", "IMX-USD",
                                   "SAND-USD", "AXS-USD", "THETA-USD", "RUNE-USD", "AAVE-USD",
                                   "EGLD-USD", "GRT-USD", "STX-USD", "MKR-USD", "KAS-USD",
                                   "FTM-USD", "FLOW-USD", "CHZ-USD", "TWT-USD", "XEC-USD",
                                   "BSV-USD", "KAVA-USD", "SNX-USD", "USDC-USD", "USDT-USD",
                                   "DAI-USD", "CRV-USD", "ENS-USD", "PEPE-USD", "LDO-USD",
                                   "RPL-USD", "DYDX-USD", "GMT-USD", "LUNC-USD", "MINA-USD",
                                   "ZEC-USD", "CAKE-USD", "BAT-USD", "ZIL-USD", "CELO-USD",
                                   "1INCH-USD", "WAVES-USD", "ANKR-USD", "COMP-USD", "GLMR-USD",
                                   "BAL-USD", "QNT-USD", "CRO-USD", "SKL-USD", "ENJ-USD",
                                   "XYM-USD", "NEXO-USD", "SUSHI-USD", "YFI-USD", "ALGO-USD",
                                   "BTT-USD", "DASH-USD", "ZEN-USD", "RVN-USD", "OMG-USD",
                                   "SRM-USD", "ICX-USD", "IOST-USD", "ONT-USD", "DENT-USD",
                                   "REQ-USD", "XNO-USD", "NANO-USD", "GALA-USD", "WOO-USD",
                                   "CVC-USD", "POLYX-USD", "BAND-USD", "STMX-USD", "POWR-USD",
                                   "JOE-USD", "ASTR-USD", "BORA-USD", "REEF-USD", "PLA-USD"],
            "Ações Brasileiras": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "MGLU3.SA",
                                 "ABEV3.SA", "JBSS3.SA", "WEGE3.SA", "RENT3.SA", "LREN3.SA",
                                 "BBAS3.SA", "GGBR4.SA", "BBSE3.SA", "CMIG4.SA", "CSAN3.SA",
                                 "ELET3.SA", "ELET6.SA", "B3SA3.SA", "SUZB3.SA", "RAIL3.SA",
                                 "CPLE6.SA", "HYPE3.SA", "BRFS3.SA", "NTCO3.SA", "BRKM5.SA",
                                 "UGPA3.SA", "HAPV3.SA", "GOLL4.SA", "VBBR3.SA", "ENEV3.SA",
                                 "AZUL4.SA", "CCRO3.SA", "TOTS3.SA", "TIMS3.SA", "PRIO3.SA",
                                 "RRRP3.SA", "SMTO3.SA", "ASAI3.SA", "ALPA4.SA", "CRFB3.SA",
                                 "CSNA3.SA", "MRFG3.SA", "KLBN11.SA", "ARZZ3.SA", "BEEF3.SA",
                                 "BRML3.SA", "CYRE3.SA", "EZTC3.SA", "CVCB3.SA", "PETZ3.SA",
                                 "SOMA3.SA", "DXCO3.SA", "VIIA3.SA", "YDUQ3.SA", "COGN3.SA",
                                 "LWSA3.SA", "BPAC11.SA", "PARD3.SA", "VIVT3.SA", "TAEE11.SA",
                                 "TRPL4.SA", "CEAB3.SA", "ENBR3.SA", "EQTL3.SA", "FLRY3.SA",
                                 "NEOE3.SA", "ALSO3.SA", "PCAR3.SA", "MRVE3.SA", "MULT3.SA",
                                 "MEAL3.SA", "MOVI3.SA", "CASH3.SA", "BRSR6.SA", "CPFE3.SA",
                                 "EMBR3.SA", "SANB11.SA", "SEQL3.SA", "RECV3.SA", "PNVL3.SA",
                                 "TEND3.SA", "BMOB3.SA", "POSI3.SA", "IGTI11.SA", "SMFT3.SA",
                                 "ENGI11.SA", "CMIN3.SA", "CEEE6.SA", "MDIA3.SA", "USIM5.SA",
                                 "BRAP4.SA", "OIBR3.SA", "OIBR4.SA", "SAPR11.SA", "TRIS3.SA",
                                 "VIVA3.SA", "LVTC3.SA", "AGRO3.SA", "AMER3.SA", "CAML3.SA",
                                 "JALL3.SA", "FESA4.SA", "DEXP3.SA", "ESPA3.SA", "AERI3.SA",
                                 "SHOW3.SA", "SOJA3.SA", "MDNE3.SA", "TTEN3.SA", "MATD3.SA",
                                 "ALLD3.SA", "RDOR3.SA", "LOGG3.SA", "GRND3.SA", "CBEE3.SA",
                                 "RAPT4.SA", "FHER3.SA", "CEGR3.SA", "JHSF3.SA", "DIRR3.SA",
                                 "INEP3.SA", "INEP4.SA", "VULC3.SA", "PRNR3.SA", "PTBL3.SA",
                                 "BLAU3.SA", "CTNM4.SA", "CATA3.SA", "BTTL3.SA", "GSHP3.SA",
                                 "CEBR6.SA", "LIPR3.SA", "FRAS3.SA", "MGEL3.SA", "MBLY3.SA",
                                 "PDGR3.SA", "HBRE3.SA", "RCSL3.SA", "RCSL4.SA", "VSPT3.SA",
                                 "APER3.SA", "MTRE3.SA", "TRAD3.SA", "TFCO4.SA", "OPCT3.SA",
                                 "PNVL4.SA", "IGBR3.SA", "BMIN3.SA", "LEVE3.SA", "TECN3.SA",
                                 "TASA3.SA", "TASA4.SA", "MODL3.SA", "MODL11.SA", "MODL4.SA",
                                 "SNSY5.SA", "IGSN3.SA", "RSID3.SA", "CGRA4.SA", "CGRA3.SA",
                                 "MNPR3.SA", "CSAB3.SA", "CSAB4.SA", "VIVA3.SA", "BRSR3.SA",
                                 "GEPA3.SA", "GEPA4.SA", "GSFI3.SA", "BDLL4.SA", "BDLL3.SA",
                                 "TELB3.SA", "TELB4.SA", "TEKA4.SA", "TEKA3.SA", "FRIO3.SA",
                                 "RANI3.SA", "UNIP6.SA", "UNIP5.SA", "UNIP3.SA", "MGSA3.SA",
                                 "MTSA4.SA", "INEP3.SA", "MTIG4.SA", "HOOT4.SA", "MTIG3.SA",
                                 "RSUL4.SA", "FNCN3.SA", "DTCY3.SA", "DEXP4.SA", "SNSY3.SA",
                                 "CTSA4.SA", "CTSA3.SA", "CTSA8.SA", "PMAM3.SA", "VITT3.SA",
                                 "EVEN3.SA", "GFSA3.SA", "CSED3.SA", "ORVR3.SA", "VITT3.SA",
                                 "BRGE11.SA", "BRGE12.SA", "BRGE3.SA", "BRGE5.SA", "BRGE6.SA",
                                 "RNEW11.SA", "RNEW3.SA", "RNEW4.SA", "LOGN3.SA", "BAZA3.SA",
                                 "POMO3.SA", "POMO4.SA", "CGAS3.SA", "CGAS5.SA", "TEKA4.SA",
                                 "BRIT3.SA", "MERC4.SA", "MERC3.SA", "AFLT3.SA", "PATI3.SA",
                                 "PATI4.SA", "MTIG3.SA", "APTI4.SA", "RPAD3.SA", "RPAD5.SA",
                                 "RPAD6.SA", "INEP4.SA", "SNSY5.SA", "MTIG4.SA", "BRIV3.SA",
                                 "BRIV4.SA", "BMEB3.SA", "BMEB4.SA", "BAHI3.SA", "BRKM3.SA",
                                 "BAHI11.SA", "BGIP3.SA", "BGIP4.SA", "NORD3.SA", "GPIV33.SA",
                                 "EMAE4.SA", "EMAE3.SA", "WLMM3.SA", "WLMM4.SA", "JOPA3.SA",
                                 "JOPA4.SA", "RPMG3.SA", "BICB3.SA", "BICB4.SA", "CTKA3.SA",
                                 "CTKA4.SA", "NINJ3.SA", "MNDL3.SA", "GPAR3.SA", "EUCA3.SA",
                                 "EUCA4.SA", "INEP3.SA", "INEP4.SA", "BRGE7.SA", "BPNM4.SA",
                                 "CTNM3.SA", "TPIS3.SA", "VLID3.SA", "BMKS3.SA", "SOJA3.SA",
                                 "KRSA3.SA", "SYNE3.SA", "IFCM3.SA", "CGRA4.SA", "CGRA3.SA",
                                 "BRSR5.SA", "ALUP11.SA", "ALUP3.SA", "ALUP4.SA", "BPRG11.SA"],
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

        # Show selected symbols
        st.write(f"**{len(symbols_list)} ativos selecionados para screening**")

    # Date range selection
    st.subheader("Date Range")

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
    st.subheader("Intervalo de Tempo (Timeframe)")
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
        "Selecione o Intervalo",
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
    st.subheader("Confirmação de Sinais")
    confirm_candles = st.number_input(
        "Candles de Confirmação",
        min_value=0,
        max_value=5,
        value=0,
        help="Número de candles consecutivos para confirmar o sinal"
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
    st.subheader("Stop Loss")
    stop_options = {
        "Stop Justo (2.0x ATR)": "Stop_Justo",
        "Stop Balanceado (2.5x ATR)": "Stop_Balanceado", 
        "Stop Largo (3.5x ATR)": "Stop_Largo"
    }

    selected_stop_display = st.selectbox(
        "Selecione o tipo de Stop Loss",
        list(stop_options.keys()),
        index=0
    )
    selected_stop = stop_options[selected_stop_display]

    # Trading direction configuration
    st.subheader("Direção de Operação")

    trading_direction = st.selectbox(
        "Escolha a direção das operações:",
        ["Ambos (Compra e Venda)", "Apenas Comprado", "Apenas Vendido"],
        index=0,
        help="Selecione se deseja operar apenas comprado, apenas vendido, ou ambas direções"
    )

    # Exit criteria configuration
    st.subheader("Critérios de Saída Personalizados")

    exit_criteria = st.selectbox(
        "Tipo de Saída",
        ["Mudança de Estado", "Stop Loss", "Alvo Fixo", "Tempo", "Média Móvel"],
        index=0,
        help="Escolha como calcular a saída das posições"
    )

    # Optimization option
    optimize_params = st.checkbox(
        "🎯 Otimizar Parâmetros",
        value=False,
        help="Testa diferentes combinações de parâmetros para encontrar o melhor retorno"
    )

    # Exit on state change option
    exit_on_state_change = st.checkbox(
        "🚪 Sair com Mudança de Estado",
        value=True,
        help="Fecha a operação automaticamente em caso de mudança de estado (buy -> sell ou sell -> buy)"
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
                target_range = st.multiselect(
                    "Alvos a Testar (%)",
                    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0],
                    default=[2.0, 3.0, 4.0, 5.0]
                )
            with col2:
                stop_range = st.multiselect(
                    "Stops a Testar (%)",
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
                    default=[1.0, 2.0, 3.0]
                )
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
            st.info("🔍 Modo Otimização: Testará de 1 a 10 candles")
            max_candles = st.number_input(
                "Máximo de candles a testar",
                min_value=5,
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
            ma_range = st.multiselect(
                "Períodos de MM a Testar",
                [10, 20, 30, 50, 60, 70, 80, 90, 100],
                default=[10, 20, 50]
            )
            exit_params['ma_range'] = ma_range

    # Analyze button
    analyze_button = st.button("🔍 Analisar", type="primary", use_container_width=True)

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
                        try:
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
                        except (IndexError, KeyError):
                            continue

                    # Get current and previous states for screening
                    current_signal = df_temp['Signal'].iloc[-1] if len(df_temp) > 0 else 'Stay Out'
                    previous_signal = df_temp['Signal'].iloc[-2] if len(df_temp) > 1 else 'Stay Out'
                    state_change = current_signal != previous_signal
                    current_price = df_temp['close'].iloc[-1] if len(df_temp) > 0 else 0

                    screening_results.append({
                        'symbol': current_symbol,
                        'status': 'OK',
                        'current_state': current_signal,
                        'previous_state': previous_signal,
                        'state_change': state_change,
                        'current_price': f"{current_price:.2f}"
                    })

                except Exception as e:
                    screening_results.append({
                        'symbol': current_symbol,
                        'status': f'Erro: {str(e)}',
                        'current_state': 'N/A',
                        'previous_state': 'N/A',
                        'state_change': False,
                        'current_price': 'N/A'
                    })

            # Update progress to 100%
            progress_bar.progress(100)
            status_text.text("Análise de screening concluída!")

            # Display screening results
            st.header("📊 Resultados do Screening")

            # Convert to DataFrame for display
            results_df = pd.DataFrame(screening_results)

            # Filter for state changes
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

            # Display all results in expandable section
            with st.expander("📋 Todos os Resultados", expanded=False):
                st.dataframe(results_df, use_container_width=True)

        else:
            # Individual analysis mode
            status_text.text(f"Baixando dados para {symbol}...")
            
            # Convert dates to strings
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Download data
            df = yf.download(symbol, start=start_str, end=end_str, interval=interval)

            if df is None or df.empty:
                st.error(f"Não foi possível obter dados para {symbol}. Verifique o ticker e tente novamente.")
                st.stop()

            # Handle multi-level columns if present
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df = df.xs(symbol, level='Ticker', axis=1, drop_level=True)

            # Ensure we have the required columns
            df.reset_index(inplace=True)
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

            progress_bar.progress(25)
            status_text.text("Calculando indicadores técnicos...")

            # Calculate moving averages
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

            # ATR calculation for stop loss
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift())
            df['low_close'] = np.abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['ATR_14'] = df['true_range'].rolling(window=14).mean()

            # Calculate different stop loss levels
            df['Stop_Justo'] = df['ATR_14'] * 2.0
            df['Stop_Balanceado'] = df['ATR_14'] * 2.5
            df['Stop_Largo'] = df['ATR_14'] * 3.5

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
                except (IndexError, KeyError):
                    continue

            progress_bar.progress(75)
            status_text.text("Calculando estatísticas...")

            # Rest of the individual analysis code would go here
            st.success("Análise concluída!")
            st.write("Individual analysis functionality to be completed...")

    except Exception as e:
        st.error(f"Erro durante a análise: {str(e)}")
    finally:
        progress_bar.progress(100)
        status_text.text("Análise finalizada!")

else:
    st.info("👆 Configure os parâmetros na barra lateral e clique em 'Analisar' para começar.")