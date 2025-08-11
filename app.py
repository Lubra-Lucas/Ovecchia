import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from contextlib import contextmanager

# Importar MT5 com tratamento de erro
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    st.warning("⚠️ MetaTrader5 não disponível. Apenas yfinance será usado.")

warnings.filterwarnings('ignore')

# --- MT5 Integration Functions ---
if MT5_AVAILABLE:
    # Mapa de timeframes MT5
    TIMEFRAME_MAP = {
        mt5.TIMEFRAME_M1: "1 Minuto",
        mt5.TIMEFRAME_M5: "5 Minutos",
        mt5.TIMEFRAME_M15: "15 Minutos",
        mt5.TIMEFRAME_M30: "30 Minutos",
        mt5.TIMEFRAME_H1: "1 Hora",
        mt5.TIMEFRAME_H4: "4 Horas",
        mt5.TIMEFRAME_D1: "Diário",
        mt5.TIMEFRAME_W1: "Semanal",
        mt5.TIMEFRAME_MN1: "Mensal",
    }

    # Mapeamento reverso para UI
    TIMEFRAME_OPTIONS = {
        "1 Minuto": mt5.TIMEFRAME_M1,
        "5 Minutos": mt5.TIMEFRAME_M5,
        "15 Minutos": mt5.TIMEFRAME_M15,
        "30 Minutos": mt5.TIMEFRAME_M30,
        "1 Hora": mt5.TIMEFRAME_H1,
        "4 Horas": mt5.TIMEFRAME_H4,
        "Diário": mt5.TIMEFRAME_D1,
        "Semanal": mt5.TIMEFRAME_W1,
        "Mensal": mt5.TIMEFRAME_MN1,
    }

    @contextmanager
    def mt5_connection():
        """Garante init/shutdown do MT5 com tratamento de erro."""
        if not mt5.initialize():
            raise RuntimeError(f"Falha ao inicializar MT5: {mt5.last_error()}")
        try:
            yield
        finally:
            mt5.shutdown()

    def _normalize_df(rates) -> pd.DataFrame:
        """Converte retorno do MT5 em DataFrame padronizado."""
        if rates is None or len(rates) == 0:
            raise ValueError("Nenhum dado retornado pelo MT5.")
        df = pd.DataFrame(rates)
        # converter timestamp e normalizar colunas
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        # remover colunas não essenciais (se existirem)
        for col in ["spread", "tick_volume", "real_volume"]:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        # garantir tipos numéricos
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # ordenar e setar índice (opcional)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df[["time","open","high","low","close","volume"]]

    def _ensure_symbol(symbol: str):
        """Garante que o símbolo está visível/assinável no MT5."""
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Símbolo '{symbol}' não encontrado no MT5.")
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Não foi possível selecionar '{symbol}' no MT5.")

    def fetch_mt5_by_range(symbol: str, timeframe: int, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Coleta candles do MT5 por intervalo de datas [start, end).
        Retorna DataFrame com colunas: time, open, high, low, close, volume.
        """
        with mt5_connection():
            _ensure_symbol(symbol)
            rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        return _normalize_df(rates)

    def fetch_mt5_last_n(symbol: str, timeframe: int, n: int = 500) -> pd.DataFrame:
        """
        Coleta os últimos N candles disponíveis no MT5.
        Útil para atualizações rápidas/tempo real.
        """
        if n <= 0:
            raise ValueError("n deve ser > 0")
        with mt5_connection():
            _ensure_symbol(symbol)
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        return _normalize_df(rates)

    def get_mt5_market_data(symbol, start_date=None, end_date=None, timeframe=mt5.TIMEFRAME_D1, mode="range", n_candles=500):
        """Função principal para coletar dados do MetaTrader 5"""
        try:
            if mode == "range" and start_date and end_date:
                # Validar que start < end
                if start_date >= end_date:
                    raise ValueError("Data inicial deve ser menor que data final")
                
                start_dt = datetime.combine(start_date, datetime.min.time()) if hasattr(start_date, 'date') else start_date
                end_dt = datetime.combine(end_date, datetime.max.time()) if hasattr(end_date, 'date') else end_date
                
                df = fetch_mt5_by_range(symbol, timeframe, start_dt, end_dt)
            else:
                # Modo "últimos N"
                df = fetch_mt5_last_n(symbol, timeframe, n_candles)

            if df.empty:
                st.error(f"Nenhum dado encontrado para {symbol} no período especificado")
                return pd.DataFrame()

            st.success(f"✅ Dados MT5 coletados: {len(df)} candles para {symbol}")
            return df

        except Exception as e:
            error_msg = f"Erro ao coletar dados MT5: {str(e)}"
            if MT5_AVAILABLE:
                mt5_error = mt5.last_error()
                if mt5_error != (0, 'Success'):
                    error_msg += f" | MT5 Error: {mt5_error}"
            st.error(error_msg)
            return pd.DataFrame()

def get_market_data(symbol, start_date, end_date, interval, data_source="yfinance", mt5_timeframe=None, mt5_mode="range", n_candles=500):
    """Função principal para coletar dados do mercado - suporta Yahoo Finance e MetaTrader 5"""
    
    if data_source == "mt5" and MT5_AVAILABLE:
        # Usar MetaTrader 5
        if mt5_timeframe is None:
            mt5_timeframe = mt5.TIMEFRAME_D1  # Default para diário
        
        # Converter string dates para datetime se necessário
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            
        return get_mt5_market_data(symbol, start_date, end_date, mt5_timeframe, mt5_mode, n_candles)
    
    else:
        # Usar Yahoo Finance (comportamento original)
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval)

            if df is None or df.empty:
                return pd.DataFrame()

            # Handle multi-level columns if present
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df = df.xs(symbol, level='Ticker', axis=1, drop_level=True)

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

            return df

        except Exception as e:
            st.error(f"Erro ao coletar dados do Yahoo Finance para {symbol}: {str(e)}")
            return pd.DataFrame()

def calcular_bollinger_bands(df, period=20):
    """Função para calcular as Bandas de Bollinger"""
    if 'close' not in df.columns:
        return None, None
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    banda_superior = sma + (2 * std)
    banda_inferior = sma - (2 * std)
    return banda_superior, banda_inferior

def display_returns_section(returns_data, criteria_name):
    """Helper function to display returns section"""
    if not returns_data.empty:
        # Get last 20 returns
        last_returns = returns_data.tail(20).copy()
        last_returns = last_returns.sort_values('exit_time', ascending=False)

        # Create mobile-friendly display for returns
        for idx, row in last_returns.iterrows():
            # Check if mobile layout should be used
            if st.session_state.get('mobile_layout', False):
                # Mobile layout - single column with cards
                return_color = "🟢" if row['return_pct'] > 0 else "🔴"
                signal_icon = "🔵" if row['signal'] == 'Buy' else "🔴"

                entry_date = row['entry_time'].strftime('%d/%m %H:%M') if hasattr(row['entry_time'], 'strftime') else str(row['entry_time'])
                exit_date = row['exit_time'].strftime('%d/%m %H:%M') if hasattr(row['exit_time'], 'strftime') else str(row['exit_time'])

                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.1rem; font-weight: bold;">{signal_icon} {row['signal']}</span>
                        <span style="font-size: 1.2rem; font-weight: bold;">{return_color} {row['return_pct']:.2f}%</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">
                        <div>📈 Entrada: {row['entry_price']:.2f} ({entry_date})</div>
                        <div>📉 Saída: {row['exit_price']:.2f} ({exit_date})</div>
                        {f'<div>🚪 {row["exit_reason"]}</div>' if 'exit_reason' in row and pd.notna(row['exit_reason']) else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Desktop layout - multi-column
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
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling with mobile improvements
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(1.5rem, 5vw, 3rem);
        font-weight: bold;
        margin-bottom: 1rem;
    }

    /* Card styling with mobile improvements */
    .metric-card {
        background: white;
        color: #333 !important;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }

    .metric-card p, .metric-card h4, .metric-card h2, .metric-card li {
        color: #333 !important;
    }

    /* Status indicators with better mobile contrast */
    .status-buy {
        background: #4CAF50;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 0.9rem;
    }

    .status-sell {
        background: #f44336;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 0.9rem;
    }

    .status-out {
        background: #757575;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 0.9rem;
    }

    /* Tab styling improvements with mobile considerations */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        min-height: 40px;
        white-space: nowrap;
        background-color: transparent;
        border-radius: 5px;
        color: #1f77b4;
        font-weight: bold;
        padding: 8px 12px;
        font-size: 0.85rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }

    /* Parameter section styling */
    .parameter-section {
        margin-bottom: 1rem;
    }

    /* Mobile responsive improvements */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
        }

        .metric-card {
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .status-buy, .status-sell, .status-out {
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 6px 8px;
            font-size: 0.75rem;
        }

        /* Ensure text is readable on mobile */
        .stMarkdown p, .stMarkdown li {
            font-size: 0.9rem;
            line-height: 1.4;
        }
    }

    /* Dark theme text fixes */
    [data-theme="dark"] .metric-card {
        background: #1e1e1e;
        color: #fff !important;
        border-left-color: #1f77b4;
    }

    [data-theme="dark"] .metric-card p, 
    [data-theme="dark"] .metric-card h4, 
    [data-theme="dark"] .metric-card h2, 
    [data-theme="dark"] .metric-card li {
        color: #fff !important;
    }
</style>""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown('<h1 class="main-title">📈 OVECCHIA TRADING - MODELO QUANT</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Sistema Avançado de Análise Técnica e Sinais de Trading</p>', unsafe_allow_html=True)

# Create main navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠 Home", "📖 Guia de Utilização", "📊 Análise Individual", "🔍 Screening Multi-Ativos", "📊 Detecção de Topos e Fundos", "🤖 Bot Telegram", "ℹ️ Sobre"])

with tab1:
    # Home page content
    st.markdown("""
    <div style="background: linear-gradient(90deg, #e3f2fd, #f3e5f5); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #1976d2; margin-bottom: 1rem;">🚀 Bem-vindo ao Sistema de Trading Quant!</h2>
        <p style="font-size: 1.2rem; color: #666;">Escolha uma das abas acima para começar sua análise profissional</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Análise Individual")
        st.markdown("""
        <div class="metric-card">
            <p><strong>🎯 Análise Detalhada de um Ativo</strong><br>
            Configure parâmetros específicos, critérios de saída personalizados e otimização de estratégias para um ativo individual.</p>
            <ul>
                <li>Gráficos interativos com sinais</li>
                <li>Múltiplos critérios de saída</li>
                <li>Otimização automática de parâmetros</li>
                <li>Análise de retornos detalhada</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🔍 Screening Multi-Ativos")
        st.markdown("""
        <div class="metric-card">
            <p><strong>📈 Monitore Múltiplos Ativos Simultaneamente</strong><br>
            Identifique rapidamente mudanças de estado em uma lista de ativos para detectar oportunidades de trading.</p>
            <ul>
                <li>Listas pré-definidas de ativos</li>
                <li>Detecção de mudanças de estado</li>
                <li>Alertas de sinais em tempo real</li>
                <li>Resumo executivo por categoria</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Detecção de Topos e Fundos")
    st.markdown("""
    <div class="metric-card">
        <p><strong>🎯 Métricas Matemáticas para identificação de extremos</strong><br>
        Detecte automaticamente possíveis topos e fundos usando variáveis matemáticas r.</p>
        <ul>
            <li>Detecção de fundos (oportunidades de compra)</li>
            <li>Detecção de topos (oportunidades de venda)</li>
            <li>Configuração personalizável de sensibilidade</li>
            <li>Análise em múltiplos timeframes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Bot Telegram - Atualizações Recentes")
    st.markdown("""
    <div class="metric-card" style="border-left: 4px solid #25D366;">
        <p><strong>🚀 Novas Funcionalidades do Bot @Ovecchia_bot</strong></p>
        <ul>
            <li><strong>📊 Análise Individual com Gráficos:</strong> Comando /analise agora gera gráficos personalizados</li>
            <li><strong>📅 Datas Personalizadas:</strong> Especifique período de análise com formato YYYY-MM-DD</li>
            <li><strong>⏰ Múltiplos Timeframes:</strong> Suporte completo para 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk</li>
            <li><strong>🎯 Estratégias Aprimoradas:</strong> Análise agressiva, balanceada e conservadora</li>
            <li><strong>📈 Gráficos Automáticos:</strong> Visualização profissional enviada como imagem</li>
        </ul>
        <p style="margin-top: 1rem; font-size: 0.9rem; color: #25D366;"><strong>💡 Exemplo:</strong> 
        <code>/analise balanceada PETR4.SA 1d 2024-01-01 2024-06-01</code></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Recursos Disponíveis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 Análises Quantitativas")
        st.markdown("""
        - Modelos baseados em variáveis estatísticas e padrões históricos
        - Algoritmos de avaliação de momentum e volatilidade
        - Faixas de oscilação dinâmica para controle de risco
        - Estratégias com múltiplos níveis de saída
        - Reconhecimento automático de zonas extremas de preço
        """)

    with col2:
        st.markdown("#### 🎯 Sinais de Trading")
        st.markdown("""
        - **Compra:** Sinais de entrada long
        - **Venda:** Sinais de entrada short
        - **Stay Out:** Ficar fora do mercado
        - **Confirmação:** Sinais consecutivos
        - **Direções:** Long, Short ou Ambos
        """)

    with col3:
        st.markdown("#### 📊 Análise de Performance")
        st.markdown("""
        - **Taxa de Acerto:** Win Rate
        - **Retorno Total:** Performance geral
        - **Maior Ganho/Perda:** Maiores variações percentuais
        - **Otimização:** Busca melhor configuração
        - **Comparação:** Múltiplos critérios
        """)

with tab2:
    # Guia de Utilização tab
    st.markdown("## 📖 Guia de Utilização - Manual Completo")
    st.markdown("Manual detalhado para utilização de todas as funcionalidades do sistema OVECCHIA TRADING")

    # Create sub-tabs for different sections
    guide_tab1, guide_tab2, guide_tab3, guide_tab4, guide_tab5 = st.tabs([
        "📊 Análise Individual", 
        "🔍 Screening Multi-Ativos", 
        "📊 Topos e Fundos", 
        "🤖 Bot Telegram",
        "⚙️ Parâmetros Gerais"
    ])

    with guide_tab1:
        st.markdown("## 📊 Guia de Utilização - Análise Individual do Ativo")

        st.markdown("### 📌 O que é a Análise Individual?")
        st.info("A Análise Individual é uma funcionalidade avançada que auxilia o usuário a identificar pontos ideais de compra e venda de um determinado ativo financeiro. Utilizando métricas matemáticas e técnicas avançadas de análise técnica, o sistema consegue captar movimentos claros de tendência, facilitando tomadas de decisão mais seguras e assertivas.")

        st.markdown("### 📌 Como funciona?")
        st.write("O sistema opera baseado em três estratégias diferentes, adaptadas para diferentes perfis de investidores:")
        st.write("• **Agressiva 🔥**: gera mais sinais, oferecendo mais oportunidades, porém com maior risco associado.")
        st.write("• **Balanceada ⚖️**: equilíbrio entre frequência de sinais e confiabilidade.")
        st.write("• **Conservadora 🛡️**: menos sinais, mas com alta confiabilidade, reduzindo a exposição ao risco.")
        
        st.write("Por exemplo, imagine que você deseja investir em PETR4.SA (Petrobras). É fundamental saber exatamente o momento certo para entrar ou sair desse ativo, protegendo seu patrimônio e maximizando lucros. Os melhores momentos são claramente exibidos no gráfico de preços com sinais coloridos:")
        st.write("• **Linha Azul 🔵**: indica ao usuário para se manter em posição comprada (apostando na alta).")
        st.write("• **Linha Vermelha 🔴**: sugere ao usuário manter posição vendida (apostando na baixa).")
        st.write("• **Linha Preta ⚫**: indica que é melhor ficar fora do mercado naquele momento.")
        
        st.write("A grande vantagem do sistema está em identificar mudanças de estado: quando o gráfico passa de vermelho para azul, é um sinal claro para entrar comprado. Da mesma forma, de azul para vermelho, é a hora de assumir uma posição vendida ou sair de uma posição comprada, aumentindo a probabilidade de capturar grandes movimentos de mercado.")
        st.write("Você também pode entrar em uma operação já em andamento e usar os pontos de Stop Loss para limitar perdas caso o mercado vá contra sua posição ou para surfar uma tendência já estabelecida, garantindo segurança e tranquilidade operacional.")

        st.markdown("### 📌 Parâmetros Essenciais")
        st.write("Para realizar a análise individual, você deverá configurar os seguintes parâmetros:")
        st.write("• **Nome do Ativo 💹**: Insira o código do ativo que deseja analisar (ex.: PETR4.SA, BTC-USD, AAPL).")
        st.write("• **Intervalo de Data 📅**: Escolha o período inicial e final da análise. Recomendamos intervalos superiores a 30 dias para maior precisão nos sinais. Atente-se às restrições históricas fornecidas pelo Yahoo Finance.")
        st.write("• **Intervalo de Tempo ⏱️**: Selecione a periodicidade desejada, como 1 minuto, 15 minutos, 1 hora, ou 1 dia, de acordo com seu perfil operacional.")
        st.write("• **Estratégia de Sinais 📈**: Selecione entre Agressiva, Balanceada ou Conservadora para ajustar o sistema ao seu apetite por risco.")
        st.write("• **Direção da Operação 🎯**: Escolha entre operar em ambas direções (comprado e vendido), somente comprado ou somente vendido.")

        st.markdown("### 📌 Critérios de Saída")
        st.write("O sistema permite que você teste estratégias variadas para saída das posições, podendo escolher entre:")
        st.write("• **Mudança de Estado 🔄**: A operação é encerrada automaticamente sempre que o estado dos sinais mudar (de compra para venda ou vice-versa).")
        st.write("• **Stop Loss 🛑**: Você define um preço limite de perda. Se o preço do ativo atingir este limite em relação ao preço de entrada, a operação é encerrada automaticamente. É um critério importante para gestão de risco eficiente.")
        st.write("• **Alvo Fixo 🎯**: Estabelece uma meta percentual de lucro e um limite percentual de perda. Ao alcançar qualquer um deles, a operação é encerrada.")
        st.write("• **Média Móvel 📉**: Neste critério, a operação é encerrada sempre que o preço cruza uma média móvel previamente configurada. A ideia é que enquanto o ativo estiver em tendência favorável, o preço estará sempre de um lado da média móvel. Caso o preço volte a cruzá-la, isso pode indicar enfraquecimento da tendência, sendo prudente sair da operação.")
        st.write("• **Tempo ⏳**: A saída ocorre após um número fixo de candles desde a entrada. Este método garante operações mais curtas e disciplinadas, reduzindo riscos de exposição prolongada. Contudo, pode limitar ganhos em tendências mais duradouras.")

        st.markdown("### 📌 Checkbox 'Sair por Mudança de Estado'")
        st.write("**🔄 Funcionalidade do Checkbox 'Sair por mudança de estado?'**")
        st.write("Este checkbox controla se as operações devem ser encerradas automaticamente quando o sistema detecta uma mudança no estado dos sinais, independentemente do critério de saída principal escolhido.")
        
        st.write("**✅ Quando ATIVADO (Marcado):**")
        st.write("• **Saída Automática**: A operação é encerrada imediatamente quando o estado muda (ex: de Buy para Sell, de Sell para Stay Out, etc.)")
        st.write("• **Prioridade Máxima**: A mudança de estado tem precedência sobre outros critérios de saída")
        st.write("• **Maior Segurança**: Evita manter posições quando o sistema já indica mudança de tendência")
        st.write("• **Operações mais Curtas**: Tende a gerar operações de menor duração")
        st.write("• **Exemplo**: Se você está comprado em PETR4 e o sistema muda de 'Buy' para 'Sell', a posição é encerrada automaticamente")
        
        st.write("**❌ Quando DESATIVADO (Desmarcado):**")
        st.write("• **Ignora Mudanças**: Operações continuam ativas mesmo com mudança de estado")
        st.write("• **Critério Principal**: Apenas o critério de saída selecionado (Stop Loss, Alvo Fixo, etc.) encerra a operação")
        st.write("• **Operações mais Longas**: Permite que operações durem mais tempo")
        st.write("• **Maior Exposição**: Mantém posições mesmo quando sistema indica reversão")
        st.write("• **Exemplo**: Se você está comprado e o sistema muda para 'Sell', você permanece comprado até atingir seu stop loss ou alvo")
        
        st.write("**💡 Recomendações de Uso:**")
        st.write("• **Ative** para estratégias mais conservadoras e seguir sinais do sistema")
        st.write("• **Desative** para testar estratégias específicas de saída sem interferência dos sinais")
        st.write("• **Para iniciantes**: Recomenda-se manter ativado para maior segurança")
        st.write("• **Para testes**: Desative para avaliar puramente a eficácia do critério de saída escolhido")

        st.markdown("### 📌 Integração com MetaTrader 5")
        if MT5_AVAILABLE:
            st.write("**🎯 MetaTrader 5 Integrado**")
            st.write("O sistema agora suporta coleta de dados diretamente do MetaTrader 5:")
            st.write("• **Requisitos**: Terminal MT5 instalado e logado em conta válida")
            st.write("• **Símbolos**: Use códigos específicos do seu corretor (ex: BTCUSD-T, WIN$, WDO$)")
            st.write("• **Timeframes**: Todos os timeframes padrão do MT5 disponíveis")
            st.write("• **Modos de coleta**: Intervalo de datas ou últimos N candles")
            st.write("• **Compatibilidade**: Funciona em análise individual, screening e topos/fundos")
            st.success("✅ **MetaTrader 5 detectado e disponível para uso!**")
        else:
            st.write("**⚠️ MetaTrader 5 Não Disponível**")
            st.write("Para usar a integração com MT5, é necessário:")
            st.write("• Instalar a biblioteca: `pip install MetaTrader5`")
            st.write("• Ter o terminal MT5 instalado na mesma máquina")
            st.write("• Terminal deve estar aberto e logado em conta válida")
        
        st.markdown("### 📌 Funcionalidade de Otimização")
        st.write("**🎯 Otimização Automática de Parâmetros**")
        st.write("O sistema oferece uma funcionalidade única de otimização automática que testa diferentes configurações para encontrar os melhores parâmetros para o ativo e período selecionados:")
        st.write("• **Teste Automático**: O sistema testa múltiplas combinações de parâmetros automaticamente")
        st.write("• **Comparação Detalhada**: Visualize uma tabela comparativa com todos os resultados testados")
        st.write("• **Melhor Configuração**: Identifica automaticamente a configuração que gerou o melhor retorno total")
        st.write("• **Múltiplas Métricas**: Avalia retorno total, retorno médio, taxa de acerto e número de operações")
        st.info("💡 **Dica**: Use a otimização para descobrir qual critério de saída funciona melhor para cada ativo específico!")

        st.markdown("### 📌 Resumo")
        st.success("Utilizar a análise individual corretamente maximiza suas chances de sucesso no mercado financeiro. Explore diferentes estratégias, teste os critérios de saída disponíveis e utilize os gráficos com sinais para tomar decisões seguras e bem fundamentadas. A combinação correta de todos esses elementos é essencial para alcançar resultados consistentes e sustentáveis em suas operações.")

    with guide_tab2:
        st.markdown("## 🔍 Guia de Utilização - Screening Multi-Ativos")

        st.markdown("### 📌 O que é o Screening?")
        st.info("O Screening Multi-Ativos é uma ferramenta poderosa que permite monitorar simultaneamente múltiplos ativos financeiros, identificando rapidamente mudanças de estado nos sinais de trading. É ideal para quem gerencia carteiras diversificadas ou quer identificar oportunidades em diferentes mercados ao mesmo tempo.")

        st.markdown("### 📌 Como Funciona?")
        st.write("O sistema aplica a mesma metodologia da análise individual, mas de forma simultânea em uma lista de ativos:")
        st.write("• **Análise Simultânea**: Processa múltiplos ativos de uma só vez")
        st.write("• **Detecção de Mudanças**: Identifica automaticamente quando um ativo muda de estado (ex: de 'Stay Out' para 'Buy')")
        st.write("• **Alertas Visuais**: Destaca ativos com mudanças recentes de estado")
        st.write("• **Resumo Executivo**: Apresenta estatísticas gerais da análise")

        st.markdown("### 📌 Listas Pré-Definidas")
        st.write("O sistema oferece listas curadas de ativos para facilitar sua análise:")
        st.write("• **🪙 Criptomoedas**: BTC-USD, ETH-USD, BNB-USD, ADA-USD, XRP-USD e mais")
        st.write("• **🇧🇷 Ações Brasileiras**: PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA e mais")
        st.write("• **🇺🇸 Ações Americanas**: AAPL, GOOGL, MSFT, AMZN, TSLA e mais")
        st.write("• **💱 Pares de Forex**: EURUSD=X, GBPUSD=X, USDJPY=X e mais")
        st.write("• **📦 Commodities**: GC=F (Ouro), SI=F (Prata), CL=F (Petróleo) e mais")
        st.info("💡 **Lista Customizada**: Você também pode criar sua própria lista inserindo os tickers desejados.")

        st.markdown("### 📌 Configurações do Screening")
        st.write("Parâmetros principais para configurar o screening:")
        st.write("• **📅 Período de Análise**: Defina o intervalo de datas para análise (padrão: últimos 30 dias)")
        st.write("• **⏱️ Timeframe**: Escolha o intervalo temporal (recomendado: 1 dia para screening)")
        st.write("• **📈 Estratégia**: Selecione entre Agressiva, Balanceada ou Conservadora")

        st.markdown("### 📌 Interpretando os Resultados")
        st.write("**🚨 Alertas de Mudança de Estado**")
        st.write("O screening destaca ativos que mudaram de estado recentemente:")
        st.write("• **🟢 Para Compra**: Ativos que mudaram para sinal de compra")
        st.write("• **🔴 Para Venda**: Ativos que mudaram para sinal de venda")
        st.write("• **⚫ Para Fora**: Ativos que mudaram para 'stay out'")
        
        st.write("**📊 Resumo Geral**")
        st.write("• **Total de Ativos**: Quantidade total analisada")
        st.write("• **Análises Bem-sucedidas**: Ativos processados sem erro")
        st.write("• **Sinais Atuais**: Distribuição dos sinais por tipo")

        st.markdown("### 📌 Melhores Práticas")
        st.write("• **🕐 Frequência**: Execute o screening diariamente para capturar mudanças recentes")
        st.write("• **📋 Listas Focadas**: Use listas específicas por categoria para análises mais direcionadas")
        st.write("• **🔍 Acompanhamento**: Monitore ativos que mudaram de estado para oportunidades")
        st.write("• **⚖️ Estratégia Balanceada**: Recomendada para screening geral")
        st.write("• **📊 Análise Complementar**: Use a análise individual para estudar ativos identificados no screening")

    with guide_tab3:
        st.markdown("## 📊 Guia de Utilização - Detecção de Topos e Fundos")

        st.markdown("### 📌 O que são Detecções Quantitativas de Topos e Fundos?")
        st.info("A Detecção Quantitativa de Topos e Fundos é uma funcionalidade especializada que utiliza métricas matemáticas e quantitativas para identificar potenciais pontos de reversão de preço. Este método aplica rigor analítico para capturar momentos em que o comportamento do mercado está anômalo em relação às suas oscilações esperadas.")

        st.markdown("### 📌 Como Funciona?")
        st.write("O sistema se baseia em métricas quantitativas:")
        st.write("• **📊 Análise de Desvios**: Utilização de desvios padrões para detectar anomalias")
        st.write("• **🟢 Detecção de Excesso de Venda**: Identificado quando métricas cruzam limites inferiores")
        st.write("• **🔴 Detecção de Excesso de Compra**: Observado quando métricas ultrapassam limites superiores")
        st.write("• **📏 Medição da Desvio**: Calcula a magnitude do desvio em relação à média esperada")

        st.markdown("### 📌 Sinais Gerados")
        st.write("**🟢 Possível Fundo (Oportunidade de Compra)**")
        st.write("Quando as variáveis do ativo indicam excesso de venda:")
        st.write("• O ativo encontra-se subvalorizado em relação à média")
        st.write("• Potencial de elevação dos preços a partir do estado atual")
        st.write("• Oportunidade para apostas compradas")
        st.write("• Maior desvio = maior potencial de correção")
        
        st.write("**🔴 Possível Topo (Oportunidade de Venda)**")
        st.write("Quando há sinais de excesso de compra:")
        st.write("• O ativo é considerado supervalorizado")
        st.write("• Potencial de queda dos preços a partir do estado atual")
        st.write("• Oportunidade de ações de venda ou desligamento de posições compradas")
        st.write("• Maior desvio = maior potencial de correção")

        st.markdown("### 📌 Configurações Disponíveis")
        st.write("• **📋 Listas de Ativos**: Mesmas opções do screening (Criptos, Ações BR/US, Forex, Commodities)")
        st.write("• **📅 Período de Análise**: Configure o intervalo de datas desejado")
        st.write("• **⏱️ Timeframe**: Recomendado usar 1h, 4h, 1d ou 1wk para melhor precisão")
        st.write("• **🎯 Sensibilidade**: Sistema usa parâmetros fixos otimizados para detectar anomalias")

        st.markdown("### 📌 Interpretando o Desvio")
        st.write("**📏 Análise do Desvio Padrão**")
        st.write("A magnitude do desvio indica a força do sinal:")
        st.write("• **0% - 1%**: Sinal fraco, correção menos provável")
        st.write("• **1% - 3%**: Sinal moderado, probabilidade de correção")
        st.write("• **3% - 5%**: Sinal forte, correção mais provável")
        st.write("• **Acima de 5%**: Sinal muito forte, alta probabilidade de correção")
        st.info("💡 **Regra Geral**: Quanto maior o desvio, maior a probabilidade de correção, mas também maior o risco.")

        st.markdown("### 📌 Estratégias de Uso")
        st.write("**📈 Para Operações de Compra (Excesso de Venda)**")
        st.write("• Espere até que métricas indiquem que o ativo está em território de venda excessiva")
        st.write("• Utilize uma abordagem de entrada gradual em diferentes pontos de preço")
        st.write("• Implementar stop loss abaixo do preço mais baixo detectado")
        st.write("• Objetivo: Retorno à média esperada de comportamento")
        
        st.write("**📉 Para Operações de Venda (Excesso de Compra)**")
        st.write("• Aguarde até que o ativo esteja em território de compra excessiva")
        st.write("• Recomenda-se encerrar posições longas")
        st.write("• Opte por vendas curtas se o mercado permitir")
        st.write("• Objetivo: Retorno à média esperada de comportamento")

        st.markdown("### 📌 Limitações e Cuidados")
        st.warning("**⚠️ Considerações Importantes**")
        st.write("• **Fortes Tendências**: Em mercados com tendências marcantes, o ativo pode permanecer desviado da média por períodos prolongados")
        st.write("• **Confirmação**: Importante validar sinais com indicadores adicionais")
        st.write("• **Gestão de Risco**: Sempre utilize stop loss, mesmo em sinais 'muito fortes'")
        st.write("• **Volatilidade**: Em mercados voláteis, sinais podem ser menos confiáveis")
        st.write("• **Volume**: Verificar volume de negociações para suporte adicional aos sinais")

    with guide_tab4:
        st.markdown("## 🤖 Guia de Utilização - Bot Telegram")

        st.markdown("### 📌 O que é o Bot Telegram?")
        st.info("O Bot Telegram @Ovecchia_bot é uma extensão do sistema que permite acesso às funcionalidades principais diretamente pelo Telegram, oferecendo análises rápidas e alertas personalizados onde quer que você esteja.")

        st.markdown("### 📌 Como Começar a Usar")
        st.write("**🚀 Passos Iniciais**")
        st.write("1. **Abra o Telegram** no seu dispositivo")
        st.write("2. **Procure por**: `@Ovecchia_bot`")
        st.write("3. **Clique em 'Iniciar'** ou digite `/start`")
        st.write("4. **Pronto!** O bot responderá com as opções disponíveis")

        st.markdown("### 📌 Comandos Disponíveis")
        st.write("**📋 Lista Completa de Comandos**")
        st.write("• **/start** - Iniciar o bot e ver mensagem de boas-vindas")
        st.write("• **/analise** - Análise individual com gráfico personalizado")
        st.write("• **/screening** - Screening de múltiplos ativos")
        st.write("• **/topos_fundos** - Detectar topos e fundos")
        st.write("• **/status** - Ver status atual do bot")
        st.write("• **/restart** - Reiniciar o bot (em caso de problemas)")
        st.write("• **/help** - Ajuda detalhada com todos os comandos")

        st.markdown("### 📌 Comando /analise - Análise Individual")
        st.write("**📊 Sintaxe Completa**")
        st.code("/analise [estrategia] [ativo] [timeframe] [data_inicio] [data_fim]")
        
        st.write("**📝 Parâmetros**")
        st.write("• **estrategia**: agressiva, balanceada ou conservadora")
        st.write("• **ativo**: ticker do ativo (ex: PETR4.SA, BTC-USD, AAPL)")
        st.write("• **timeframe**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk")
        st.write("• **data_inicio**: formato YYYY-MM-DD (opcional)")
        st.write("• **data_fim**: formato YYYY-MM-DD (opcional)")
        
        st.write("**💡 Exemplos**")
        st.code("/analise balanceada PETR4.SA 1d")
        st.code("/analise agressiva BTC-USD 4h 2024-01-01 2024-06-01")
        st.code("/analise conservadora AAPL 1d")
        
        st.success("**📈 Resultado**: O bot gerará um gráfico personalizado e enviará como imagem junto com análise detalhada")

        st.markdown("### 📌 Comando /screening - Múltiplos Ativos")
        st.write("**🔍 Sintaxe**")
        st.code("/screening [estrategia] [ativo1] [ativo2] [ativo3] ...")
        
        st.write("**💡 Exemplos**")
        st.code("/screening balanceada BTC-USD ETH-USD")
        st.code("/screening agressiva PETR4.SA VALE3.SA ITUB4.SA")
        st.code("/screening conservadora AAPL GOOGL MSFT")
        
        st.success("**📊 Resultado**: Lista mudanças de estado recentes nos ativos especificados")

        st.markdown("### 📌 Comando /topos_fundos - Extremos")
        st.write("**📊 Sintaxe**")
        st.code("/topos_fundos [ativo1] [ativo2] [ativo3] ...")
        
        st.write("**💡 Exemplos**")
        st.code("/topos_fundos PETR4.SA VALE3.SA")
        st.code("/topos_fundos BTC-USD ETH-USD BNB-USD")
        st.code("/topos_fundos AAPL GOOGL")
        
        st.success("**📈 Resultado**: Identifica possíveis topos e fundos usando Bandas de Bollinger")

        st.markdown("### 📌 Recursos Especiais do Bot")
        st.write("**🎯 Funcionalidades Exclusivas**")
        st.write("• **📊 Gráficos Automáticos**: Geração e envio automático de gráficos")
        st.write("• **⚡ Respostas Rápidas**: Análises em poucos segundos")
        st.write("• **📱 Disponibilidade 24/7**: Bot ativo 24 horas por dia")
        st.write("• **🔄 Auto-Recovery**: Sistema de restart automático em caso de falhas")
        st.write("• **📋 Validação Automática**: Verificação de parâmetros e formatos")
        st.write("• **🗂️ Limpeza Automática**: Remove arquivos temporários automaticamente")

        st.markdown("### 📌 Dicas de Uso")
        st.write("**💡 Melhores Práticas**")
        st.write("• **⏰ Timing**: Use o bot preferencialmente fora de horários de alta volatilidade")
        st.write("• **📊 Estratégias**: Comece com 'balanceada' para ter equilíbrio")
        st.write("• **⚖️ Múltiplos Ativos**: No screening, limite a 10 ativos por comando")
        st.write("• **📅 Datas**: Para análises históricas, use períodos mínimos de 30 dias")
        st.write("• **🔄 Problemas**: Se o bot não responder, use /restart")
        st.write("• **💾 Armazenamento**: Salve gráficos importantes, pois são temporários")

        st.markdown("### 📌 Status e Troubleshooting")
        st.write("**🔧 Resolução de Problemas**")
        st.write("• **Bot não responde**: Use /restart ou aguarde alguns minutos")
        st.write("• **Erro de ativo**: Verifique se o ticker está correto (ex: PETR4.SA, não PETR4)")
        st.write("• **Erro de data**: Use formato YYYY-MM-DD (ex: 2024-01-15)")
        st.write("• **Timeframe inválido**: Use apenas: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk")
        st.write("• **Comando não reconhecido**: Use /help para ver lista completa")
        
        st.write("**📊 Verificar Status**")
        st.write("• Use `/status` para verificar se o bot está funcionando")
        st.write("• Resposta esperada: informações sobre tempo online e status dos serviços")

    with guide_tab5:
        st.markdown("## ⚙️ Guia de Parâmetros Gerais")

        st.markdown("### 📌 Tipos de Ativos Suportados")
        st.write("**📊 Formato de Tickers por Categoria**")
        st.write("• **🪙 Criptomoedas**: Use sufixo -USD (ex: BTC-USD, ETH-USD)")
        st.write("• **🇧🇷 Ações Brasileiras**: Use sufixo .SA (ex: PETR4.SA, VALE3.SA)")
        st.write("• **🇺🇸 Ações Americanas**: Use ticker direto (ex: AAPL, GOOGL, MSFT)")
        st.write("• **💱 Forex**: Use sufixo =X (ex: EURUSD=X, GBPUSD=X)")
        st.write("• **📦 Commodities**: Use sufixo =F (ex: GC=F para ouro, CL=F para petróleo)")

        st.markdown("### 📌 Intervalos de Tempo (Timeframes)")
        st.write("**⏱️ Timeframes Disponíveis e Recomendações**")
        st.write("• **1m, 2m, 5m**: Scalping e day trading (dados limitados a 7 dias no Yahoo Finance)")
        st.write("• **15m, 30m**: Day trading e swing trading intraday")
        st.write("• **60m, 90m**: Swing trading de curto prazo")
        st.write("• **4h**: Swing trading de médio prazo")
        st.write("• **1d**: Position trading e análises de médio/longo prazo (mais recomendado)")
        st.write("• **5d, 1wk**: Análises de longo prazo")
        st.write("• **1mo, 3mo**: Análises macro e tendências de muito longo prazo")
        st.info("💡 **Recomendação**: Para análises gerais, use 1d (1 dia) para melhor equilíbrio entre dados históricos e precisão.")

        st.markdown("### 📌 Estratégias de Trading")
        st.write("**🎯 Perfis de Estratégia**")
        
        st.write("**🔥 Estratégia Agressiva**")
        st.write("• Algoritmo calibrado para maior sensibilidade")
        st.write("• Gera mais sinais de entrada")
        st.write("• Maior frequência de operações")
        st.write("• Maior potencial de lucro, mas também maior risco")
        st.write("• Ideal para: Traders experientes, mercados com tendência clara")
        
        st.write("**⚖️ Estratégia Balanceada**")
        st.write("• Configuração otimizada para equilíbrio")
        st.write("• Equilíbrio entre frequência e confiabilidade")
        st.write("• Recomendada para maioria dos usuários")
        st.write("• Boa relação risco/retorno")
        st.write("• Ideal para: Investidores intermediários, carteiras diversificadas")
        
        st.write("**🛡️ Estratégia Conservadora**")
        st.write("• Parâmetros ajustados para maior segurança")
        st.write("• Menos sinais, mas mais confiáveis")
        st.write("• Menor frequência de operações")
        st.write("• Foco em preservação de capital")
        st.write("• Ideal para: Investidores iniciantes, mercados voláteis")

        st.markdown("### 📌 Direções de Operação")
        st.write("**🎯 Tipos de Operação**")
        st.write("• **Ambos (Compra e Venda)**: Opera em ambas direções, maximiza oportunidades")
        st.write("• **Apenas Comprado**: Só opera na alta (long only), ideal para mercados em alta")
        st.write("• **Apenas Vendido**: Só opera na baixa (short only), ideal para mercados em queda")
        st.warning("⚠️ **Importante**: Nem todos os ativos/brokers permitem operações vendidas (short). Verifique as regras do seu provedor.")

        st.markdown("### 📌 Tipos de Stop Loss")
        st.write("**🛡️ Sistema de Stop Loss Baseado em Volatilidade**")
        st.write("O sistema oferece três tipos de stop loss calculados dinamicamente com base na volatilidade do ativo:")
        
        st.write("• **Stop Justo**: Nível mais próximo ao preço (mais proteção, saídas mais frequentes)")
        st.write("• **Stop Balanceado**: Nível intermediário (equilíbrio entre proteção e permanência)")
        st.write("• **Stop Largo**: Nível mais distante (menos saídas por ruído, perdas maiores quando ocorrem)")
        
        st.write("**📊 Como Funciona**")
        st.write("• O sistema calcula automaticamente os níveis com base na volatilidade atual")
        st.write("• Stop se adapta automaticamente às condições de mercado")
        st.write("• Cada tipo oferece um perfil diferente de risco/retorno")
        st.write("• Recomenda-se testar diferentes tipos para encontrar o ideal para seu perfil")

        st.markdown("### 📌 Limitações dos Dados")
        st.warning("**⚠️ Limitações do Yahoo Finance**")
        st.write("• **Dados Intraday**: Timeframes menores que 1 dia têm limite de 7 dias históricos")
        st.write("• **Fins de Semana**: Mercados fechados podem afetar dados em tempo real")
        st.write("• **Feriados**: Dados podem estar indisponíveis em feriados locais")
        st.write("• **Ativos Descontinuados**: Alguns tickers podem não ter dados atualizados")
        st.write("• **Splits/Dividendos**: Podem causar descontinuidades nos dados históricos")
        
        st.info("**💡 Dicas para Evitar Problemas**")
        st.write("• Use timeframe 1d para análises históricas longas")
        st.write("• Verifique se o ticker está correto antes de analisar")
        st.write("• Para timeframes menores, use períodos recentes (última semana)")
        st.write("• Se encontrar erros, tente ticker alternativo ou período menor")

with tab3:
    # Individual Analysis tab
    st.markdown("## 📊 Análise Individual de Ativo")
    st.markdown("Configure os parâmetros para análise detalhada de um ativo específico")

    # Create parameter sections
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.markdown("#### 💹 Configuração de Ativo")
        
        # Seleção da fonte de dados
        if MT5_AVAILABLE:
            data_source_options = ["Yahoo Finance", "MetaTrader 5"]
            data_source = st.selectbox("Fonte de Dados", data_source_options, index=0)
            data_source_key = "yfinance" if data_source == "Yahoo Finance" else "mt5"
        else:
            data_source = "Yahoo Finance"
            data_source_key = "yfinance"
            st.info("📊 **Fonte de Dados:** Yahoo Finance")
        
        # Símbolo com exemplos específicos por fonte
        if data_source_key == "mt5":
            symbol = st.text_input(
                "Símbolo MT5",
                value="BTCUSD-T",
                help="Exemplos: BTCUSD-T, WIN$, WDO$, EURUSD-T"
            ).strip()
        else:
            symbol = st.text_input(
                "Ticker",
                value="BTC-USD",
                help="Examples: BTC-USD, PETR4.SA, AAPL, EURUSD=X"
            ).strip()

        st.markdown("#### 📅 Intervalo de Data")
        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=365)

        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Data Inicial", value=default_start, max_value=default_end)
        with col_date2:
            end_date = st.date_input("Data Final", value=default_end, min_value=start_date, max_value=default_end)

        st.markdown("#### ⏱️ Intervalo de Tempo")
        
        if data_source_key == "mt5" and MT5_AVAILABLE:
            # Opções específicas do MT5
            timeframe_display = st.selectbox("Timeframe MT5", list(TIMEFRAME_OPTIONS.keys()), index=6)  # Diário como padrão
            mt5_timeframe = TIMEFRAME_OPTIONS[timeframe_display]
            interval = "1d"  # Para compatibilidade com o resto do código
            
            # Modo de coleta MT5
            st.markdown("#### 📊 Modo de Coleta MT5")
            mt5_mode = st.radio(
                "Modo:",
                ["Intervalo de datas", "Últimos N candles"],
                index=0
            )
            
            if mt5_mode == "Últimos N candles":
                n_candles = st.number_input("Número de candles", min_value=1, max_value=10000, value=1000)
            else:
                n_candles = 500
                
        else:
            # Opções do Yahoo Finance
            interval_options = {
                "1 minute": "1m", "2 minutes": "2m", "5 minutes": "5m", "15 minutes": "15m",
                "30 minutes": "30m", "60 minutes": "60m", "90 minutes": "90m", "4 hours": "4h",
                "1 day": "1d", "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"
            }
            interval_display = st.selectbox("Intervalo", list(interval_options.keys()), index=8)
            interval = interval_options[interval_display]
            mt5_timeframe = None
            mt5_mode = "range"
            n_candles = 500

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        

        st.markdown("#### 📈 Estratégia de Sinais")
        st.markdown("""
        <div style="background: #f0f2f6; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="margin: 0; font-size: 0.85rem; color: #333;">
                <strong>ℹ️ Guia de Estratégias:</strong><br>
                • <strong>Agressivo:</strong> Maior quantidade de sinais (mais oportunidades, maior risco)<br>
                • <strong>Balanceado:</strong> Quantidade média de sinais (equilíbrio entre oportunidade e confiabilidade)<br>
                • <strong>Conservador:</strong> Poucos sinais, mas mais confiáveis (menor risco, menos oportunidades)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        strategy_type = st.radio(
            "Tipo de Estratégia:",
            ["Balanceado", "Agressivo", "Conservador"],
            index=0,
            help="Escolha a estratégia baseada no seu perfil de risco e frequência desejada de sinais"
        )
        
        # Definir parâmetros baseado na estratégia selecionada
        if strategy_type == "Agressivo":
            sma_short = 10
            sma_long = 21
        elif strategy_type == "Conservador":
            sma_short = 140
            sma_long = 200
        else:  # Balanceado
            sma_short = 60
            sma_long = 70

        st.markdown("#### 🎯 Direção de Operação")
        trading_direction = st.selectbox(
            "Direção das operações:",
            ["Ambos (Compra e Venda)", "Apenas Comprado", "Apenas Vendido"],
            index=0
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Exit criteria section
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("#### 🚪 Critérios de Saída Personalizados")

    col_exit1, col_exit2, col_exit3 = st.columns(3)

    with col_exit1:
        exit_criteria = st.selectbox(
            "Tipo de Saída",
            ["Mudança de Estado", "Stop Loss", "Alvo Fixo", "Tempo", "Média Móvel"],
            index=0
        )

    with col_exit2:
        include_state_change = st.checkbox("Sair por mudança de estado?", value=True)

    with col_exit3:
        optimize_params = st.checkbox("🎯 Otimizar Parâmetros", value=False)

    # Additional parameters based on exit criteria
    exit_params = {}

    if exit_criteria == "Stop Loss" and not optimize_params:
        exit_params['stop_type'] = st.selectbox("Tipo de Stop", ["Stop Justo", "Stop Balanceado", "Stop Largo"])
    elif exit_criteria == "Alvo Fixo" and not optimize_params:
        col_target1, col_target2 = st.columns(2)
        with col_target1:
            exit_params['target_pct'] = st.number_input("Alvo (%)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
        with col_target2:
            exit_params['stop_loss_pct'] = st.number_input("Stop Loss (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    elif exit_criteria == "Tempo" and not optimize_params:
        exit_params['time_candles'] = st.number_input("Candles após entrada", min_value=1, max_value=1000, value=10, step=1)
    elif exit_criteria == "Média Móvel" and not optimize_params:
        exit_params['ma_period'] = st.number_input("Período da MM", min_value=5, max_value=200, value=20, step=5)

    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis button
    analyze_button_individual = st.button("🚀 INICIAR ANÁLISE INDIVIDUAL", type="primary", use_container_width=True)

    # Analysis logic (same as before but only for individual analysis)
    if analyze_button_individual:
        if not symbol:
            st.error("Por favor entre com um ticker válido.")
            st.stop()

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Fetch data
            status_text.text("Coletando dados de mercado...")
            progress_bar.progress(20)

            # Convert dates to strings
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Download data using appropriate API
            mt5_mode_key = "last_n" if data_source_key == "mt5" and mt5_mode == "Últimos N candles" else "range"
            df = get_market_data(
                symbol, 
                start_str, 
                end_str, 
                interval, 
                data_source=data_source_key,
                mt5_timeframe=mt5_timeframe,
                mt5_mode=mt5_mode_key,
                n_candles=n_candles
            )

            if df is None or df.empty:
                st.error(f"Sem data encontrada para '{symbol}' nesse período de tempo.")
                st.stop()

            progress_bar.progress(40)
            status_text.text("Processando indicadores...")

            # Data preprocessing
            symbol_label = symbol.replace("=X", "").replace("-USD", "")

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

            # State persistence - aplicar sinal imediatamente
            df['Estado'] = 'Stay Out'

            for i in range(len(df)):
                if i == 0:
                    # Primeiro candle sempre Stay Out
                    continue

                # Estado anterior
                estado_anterior = df['Estado'].iloc[i - 1]

                # Aplicar sinal imediatamente
                sinal_atual = df['Signal'].iloc[i]
                if sinal_atual != 'Stay Out':
                    df.loc[df.index[i], 'Estado'] = sinal_atual
                else:
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
            titulo_grafico = f"OVECCHIA TRADING - {symbol_label} - Timeframe: {interval.upper()}"

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

with tab4:
    # Screening tab
    st.markdown("## 🔍 Screening de Múltiplos Ativos")
    st.info("ℹ️ **Screening Mode:** O screening focará apenas na detecção de mudanças de estado dos sinais.")

    # Screening parameters
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.markdown("#### 📊 Lista de Ativos")

        # Predefined lists
        preset_lists = {
            "Criptomoedas": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD",
                                   "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD",
                                   "TRX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
                                   "FIL-USD", "APT-USD", "ARB-USD", "NEAR-USD", "VET-USD"],
            "Ações Brasileiras": [
                "ABEV3.SA", "ALPA4.SA", "AMER3.SA", "ARZZ3.SA", "ASAI3.SA",
                "AZUL4.SA", "B3SA3.SA", "BBAS3.SA", "BBDC3.SA", "BBDC4.SA",
                "BBSE3.SA", "BEEF3.SA", "BPAC11.SA", "BPAN4.SA", "BRAP4.SA",
                "BRFS3.SA", "BRKM5.SA", "CASH3.SA", "CCRO3.SA", "CIEL3.SA",
                "CMIG4.SA", "CMIN3.SA", "COGN3.SA", "CPFE3.SA", "CPLE6.SA",
                "CRFB3.SA", "CSAN3.SA", "CSMG3.SA", "CSNA3.SA", "CVCB3.SA",
                "CYRE3.SA", "DXCO3.SA", "EGIE3.SA", "ELET3.SA", "ELET6.SA",
                "EMBR3.SA", "ENBR3.SA", "ENEV3.SA", "ENGI11.SA", "EQTL3.SA",
                "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "GOLL4.SA",
                "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ITSA4.SA",
                "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "LWSA3.SA",
                "MGLU3.SA", "MOVI3.SA", "MRFG3.SA", "MRVE3.SA", "MULT3.SA",
                "NTCO3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "PETZ3.SA",
                "POSI3.SA", "PRIO3.SA", "QUAL3.SA", "RADL3.SA", "RAIL3.SA",
                "RAIZ4.SA", "RDOR3.SA", "RENT3.SA", "SANB11.SA", "SBSP3.SA",
                "SLCE3.SA", "SMTO3.SA", "SOMA3.SA", "SUZB3.SA", "TAEE11.SA",
                "TIMS3.SA", "TOTS3.SA", "TRPL4.SA", "UGPA3.SA", "USIM5.SA",
                "VALE3.SA", "VAMO3.SA", "VBBR3.SA", "VIIA3.SA", "VIVT3.SA",
                "WEGE3.SA", "YDUQ3.SA", "ALSO3.SA", "SEQL3.SA", "SIMH3.SA",
                "TTEN3.SA", "VIVA3.SA", "WEST3.SA", "OIBR4.SA", "CMIG3.SA",
                "AESB3.SA", "NEOE3.SA", "CAML3.SA", "POMO4.SA", "GRND3.SA",
                "ODPV3.SA", "ENAT3.SA", "LOGG3.SA", "MDIA3.SA", "RECV3.SA",
                "SAPR11.SA", "SAPR4.SA", "SBFG3.SA", "TEND3.SA", "TFCO4.SA",
                "HBOR3.SA", "HBSA3.SA", "SHOW3.SA", "ESPA3.SA", "ROMI3.SA",
                "JHSF3.SA", "GUAR3.SA", "KEPL3.SA", "JSLG3.SA", "PGMN3.SA",
                "PNVL3.SA", "PTBL3.SA", "RAPT4.SA", "SEER3.SA", "WIZC3.SA"
            ],
            "Ações Americanas": [
                "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "BRK-B", "TSLA", 
                "TSM", "JPM", "WMT", "LLY", "ORCL", "V", "MA", "NFLX", "XOM", "COST", 
                "JNJ", "PLTR", "HD", "PG", "BAC", "ABBV", "KO", "CVX", "CRM", "UNH", 
                "PM", "IBM", "MS", "GS", "LIN", "INTU", "ABT", "DIS", "AXP", "MRK", 
                "MCD", "RTX", "CAT", "T", "NOW", "PEP", "UBER", "BKNG", "VZ", "TMO", 
                "ISRG", "ACN", "C", "SCHW", "GEV", "BA", "BLK", "QCOM", "TXN", "AMGN", 
                "SPGI", "ADBE", "BSX", "SYK", "ETN", "SO", "SPG", "TMUS", "NKE", "HON", 
                "MDT", "MMM", "MO", "USB", "LMT", "UPS", "UNP", "PYPL", "TGT", "DE", 
                "GILD", "CMCSA", "CHTR", "COP", "GE", "FDX", "DUK", "EMR", "DD", "NEE", 
                "SBUX", "F", "GM", "OXY", "BIIB", "CVS", "CL", "ED", "GLW", "D", 
                "PFE", "DG", "ADP", "ZTS", "BBY", "MNST", "TRV", "SLB", "ICE", "WELL", 
                "EL", "FOXA", "FOX", "KR", "PSX", "ADM", "APD", "EQIX", "CMS", "WFC", 
                "NOC", "EXC", "SYY", "AON", "MET", "AFL", "TJX", "BMY", "HAL", "STZ"
            ],
            "Pares de Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X"],
            "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "KE=F", "CC=F", "KC=F"]
        }

        selected_preset = st.selectbox(
            "Lista:",
            ["Customizada"] + list(preset_lists.keys())
        )

        if selected_preset != "Customizada":
            symbols_list = preset_lists[selected_preset]
            st.info(f"{len(symbols_list)} ativos selecionados")
        else:
            symbols_input = st.text_area(
                "Tickers (um por linha):",
                value="BTC-USD\nETH-USD\nPETR4.SA\nAAPL",
                height=100
            )
            symbols_list = [s.strip() for s in symbols_input.split('\n') if s.strip()]
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.markdown("#### 📅 Configurações de Análise")
        
        # Seleção da fonte de dados para screening
        if MT5_AVAILABLE:
            screening_data_source_options = ["Yahoo Finance", "MetaTrader 5"]
            screening_data_source = st.selectbox("Fonte de Dados", screening_data_source_options, index=0, key="screening_source")
            screening_data_source_key = "yfinance" if screening_data_source == "Yahoo Finance" else "mt5"
        else:
            screening_data_source = "Yahoo Finance"
            screening_data_source_key = "yfinance"

        # Fixed period: 2 years
        default_end_screening = datetime.now().date()
        default_start_screening = default_end_screening - timedelta(days=730)  # 2 years

        start_date_screening = default_start_screening
        end_date_screening = default_end_screening
        
        if screening_data_source_key == "mt5":
            st.info("📅 **Período:** 2 anos de dados históricos")
            st.info("⏰ **Timeframe:** Diário (MT5)")
            screening_mt5_timeframe = mt5.TIMEFRAME_D1
        else:
            st.info("📅 **Período fixo:** 2 anos de dados históricos")
            st.info("⏰ **Timeframe fixo:** 1 dia")
            screening_mt5_timeframe = None

        # Fixed interval: 1 day
        interval_screening = "1d"

        

        # Strategy selection
        st.markdown("#### 📈 Estratégia de Sinais")
        st.markdown("""
        <div style="background: #f0f2f6; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="margin: 0; font-size: 0.85rem; color: #333;">
                <strong>ℹ️ Guia de Estratégias:</strong><br>
                • <strong>Agressivo:</strong> Maior quantidade de sinais (mais oportunidades, maior risco)<br>
                • <strong>Balanceado:</strong> Quantidade média de sinais (equilíbrio entre oportunidade e confiabilidade)<br>
                • <strong>Conservador:</strong> Poucos sinais, mas mais confiáveis (menor risco, menos oportunidades)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        strategy_type_screening = st.radio(
            "Tipo de Estratégia:",
            ["Balanceado", "Agressivo", "Conservador"],
            index=0,
            key="strategy_screening",
            help="Escolha a estratégia baseada no seu perfil de risco e frequência desejada de sinais"
        )
        
        # Definir parâmetros baseado na estratégia selecionada
        if strategy_type_screening == "Agressivo":
            sma_short_screening = 10
            sma_long_screening = 21
        elif strategy_type_screening == "Conservador":
            sma_short_screening = 140
            sma_long_screening = 200
        else:  # Balanceado
            sma_short_screening = 60
            sma_long_screening = 70

        st.markdown('</div>', unsafe_allow_html=True)

    # Analysis button for screening
    analyze_button_screening = st.button("🚀 INICIAR SCREENING", type="primary", use_container_width=True)

    # Screening analysis logic
    if analyze_button_screening:
        if not symbols_list:
            st.error("Por favor selecione pelo menos um ativo para screening.")
            st.stop()

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Screening mode
            screening_results = []
            total_symbols = len(symbols_list)

            for idx, current_symbol in enumerate(symbols_list):
                status_text.text(f"Analisando {current_symbol} ({idx+1}/{total_symbols})...")
                progress_bar.progress(int((idx / total_symbols) * 100))

                try:
                    # Convert dates to strings
                    start_str = start_date_screening.strftime("%Y-%m-%d")
                    end_str = end_date_screening.strftime("%Y-%m-%d")

                    # Download data using appropriate API
                    df_temp = get_market_data(
                        current_symbol, 
                        start_str, 
                        end_str, 
                        interval_screening,
                        data_source=screening_data_source_key,
                        mt5_timeframe=screening_mt5_timeframe,
                        mt5_mode="range",
                        n_candles=1000
                    )

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

                    # Calculate indicators (simplified for screening)
                    df_temp[f'SMA_{sma_short_screening}'] = df_temp['close'].rolling(window=sma_short_screening).mean()
                    df_temp[f'SMA_{sma_long_screening}'] = df_temp['close'].rolling(window=sma_long_screening).mean()
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
                            df_temp['close'].iloc[i] > df_temp[f'SMA_{sma_short_screening}'].iloc[i]
                            and df_temp['close'].iloc[i] > df_temp[f'SMA_{sma_long_screening}'].iloc[i]
                            and rsi_up and rsl_buy
                        ):
                            df_temp.at[i, 'Signal'] = 'Buy'
                        elif (
                            df_temp['close'].iloc[i] < df_temp[f'SMA_{sma_short_screening}'].iloc[i]
                            and rsi_down and rsl_sell
                        ):
                            df_temp.at[i, 'Signal'] = 'Sell'

                    # State persistence - aplicar sinal imediatamente
                    df_temp['Estado'] = 'Stay Out'

                    for i in range(len(df_temp)):
                        if i == 0:
                            # Primeiro candle sempre Stay Out
                            continue

                        # Estado anterior
                        estado_anterior = df_temp['Estado'].iloc[i - 1]

                        # Aplicar sinal imediatamente
                        sinal_atual = df_temp['Signal'].iloc[i]
                        if sinal_atual != 'Stay Out':
                            df_temp.loc[df_temp.index[i], 'Estado'] = sinal_atual
                        else:
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
            st.success(f"✅ Screening completo para {len(symbols_list)} ativos")

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

        except Exception as e:
            st.error(f"An error occurred during screening: {str(e)}")
            st.write("Please check your inputs and try again.")

with tab5:
    # Bollinger Bands Detection tab
    st.markdown("## 📊 Detecção de Topos e Fundos")
    st.markdown("Identifique oportunidades de compra e venda baseadas em métricas matemáticas")

    # Parameters section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.markdown("#### 💹 Lista de Ativos")

        # Predefined lists for Bollinger Bands screening
        preset_lists_bb = {
            "Criptomoedas Top": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD"],
            "Ações Brasileiras": [
                "ABEV3.SA", "ALPA4.SA", "AMER3.SA", "ARZZ3.SA", "ASAI3.SA",
                "AZUL4.SA", "B3SA3.SA", "BBAS3.SA", "BBDC3.SA", "BBDC4.SA",
                "BBSE3.SA", "BEEF3.SA", "BPAC11.SA", "BPAN4.SA", "BRAP4.SA",
                "BRFS3.SA", "BRKM5.SA", "CASH3.SA", "CCRO3.SA", "CIEL3.SA",
                "CMIG4.SA", "CMIN3.SA", "COGN3.SA", "CPFE3.SA", "CPLE6.SA",
                "CRFB3.SA", "CSAN3.SA", "CSMG3.SA", "CSNA3.SA", "CVCB3.SA",
                "CYRE3.SA", "DXCO3.SA", "EGIE3.SA", "ELET3.SA", "ELET6.SA",
                "EMBR3.SA", "ENBR3.SA", "ENEV3.SA", "ENGI11.SA", "EQTL3.SA",
                "EZTC3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "GOLL4.SA",
                "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ITSA4.SA",
                "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "LWSA3.SA",
                "MGLU3.SA", "MOVI3.SA", "MRFG3.SA", "MRVE3.SA", "MULT3.SA",
                "NTCO3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "PETZ3.SA",
                "POSI3.SA", "PRIO3.SA", "QUAL3.SA", "RADL3.SA", "RAIL3.SA",
                "RAIZ4.SA", "RDOR3.SA", "RENT3.SA", "SANB11.SA", "SBSP3.SA",
                "SLCE3.SA", "SMTO3.SA", "SOMA3.SA", "SUZB3.SA", "TAEE11.SA",
                "TIMS3.SA", "TOTS3.SA", "TRPL4.SA", "UGPA3.SA", "USIM5.SA",
                "VALE3.SA", "VAMO3.SA", "VBBR3.SA", "VIIA3.SA", "VIVT3.SA",
                "WEGE3.SA", "YDUQ3.SA", "ALSO3.SA", "SEQL3.SA", "SIMH3.SA",
                "TTEN3.SA", "VIVA3.SA", "WEST3.SA", "OIBR4.SA", "CMIG3.SA",
                "AESB3.SA", "NEOE3.SA", "CAML3.SA", "POMO4.SA", "GRND3.SA",
                "ODPV3.SA", "ENAT3.SA", "LOGG3.SA", "MDIA3.SA", "RECV3.SA",
                "SAPR11.SA", "SAPR4.SA", "SBFG3.SA", "TEND3.SA", "TFCO4.SA",
                "HBOR3.SA", "HBSA3.SA", "SHOW3.SA", "ESPA3.SA", "ROMI3.SA",
                "JHSF3.SA", "GUAR3.SA", "KEPL3.SA", "JSLG3.SA", "PGMN3.SA",
                "PNVL3.SA", "PTBL3.SA", "RAPT4.SA", "SEER3.SA", "WIZC3.SA"
            ],
            "Ações Americanas": [
                "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "BRK-B", "TSLA", 
                "TSM", "JPM", "WMT", "LLY", "ORCL", "V", "MA", "NFLX", "XOM", "COST", 
                "JNJ", "PLTR", "HD", "PG", "BAC", "ABBV", "KO", "CVX", "CRM", "UNH", 
                "PM", "IBM", "MS", "GS", "LIN", "INTU", "ABT", "DIS", "AXP", "MRK", 
                "MCD", "RTX", "CAT", "T", "NOW", "PEP", "UBER", "BKNG", "VZ", "TMO", 
                "ISRG", "ACN", "C", "SCHW", "GEV", "BA", "BLK", "QCOM", "TXN", "AMGN", 
                "SPGI", "ADBE", "BSX", "SYK", "ETN", "SO", "SPG", "TMUS", "NKE", "HON", 
                "MDT", "MMM", "MO", "USB", "LMT", "UPS", "UNP", "PYPL", "TGT", "DE", 
                "GILD", "CMCSA", "CHTR", "COP", "GE", "FDX", "DUK", "EMR", "DD", "NEE", 
                "SBUX", "F", "GM", "OXY", "BIIB", "CVS", "CL", "ED", "GLW", "D", 
                "PFE", "DG", "ADP", "ZTS", "BBY", "MNST", "TRV", "SLB", "ICE", "WELL", 
                "EL", "FOXA", "FOX", "KR", "PSX", "ADM", "APD", "EQIX", "CMS", "WFC", 
                "NOC", "EXC", "SYY", "AON", "MET", "AFL", "TJX", "BMY", "HAL", "STZ"
            ],
            "Forex Principais": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"],
            "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]
        }

        selected_preset_bb = st.selectbox(
            "Lista:",
            ["Customizada"] + list(preset_lists_bb.keys()),
            key="preset_bb"
        )

        if selected_preset_bb != "Customizada":
            symbols_list_bb = preset_lists_bb[selected_preset_bb]
            st.info(f"{len(symbols_list_bb)} ativos selecionados")
        else:
            symbols_input_bb = st.text_area(
                "Tickers (um por linha):",
                value="BTC-USD\nETH-USD\nPETR4.SA\nAAPL",
                height=100,
                key="symbols_bb"
            )
            symbols_list_bb = [s.strip() for s in symbols_input_bb.split('\n') if s.strip()]
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.markdown("#### 📅 Configurações de Análise")
        
        # Seleção da fonte de dados para topos e fundos
        if MT5_AVAILABLE:
            bb_data_source_options = ["Yahoo Finance", "MetaTrader 5"]
            bb_data_source = st.selectbox("Fonte de Dados", bb_data_source_options, index=0, key="bb_source")
            bb_data_source_key = "yfinance" if bb_data_source == "Yahoo Finance" else "mt5"
        else:
            bb_data_source = "Yahoo Finance"
            bb_data_source_key = "yfinance"

        # Fixed period: 2 years
        default_end_bb = datetime.now().date()
        default_start_bb = default_end_bb - timedelta(days=730)  # 2 years

        start_date_bb = default_start_bb
        end_date_bb = default_end_bb
        
        if bb_data_source_key == "mt5":
            st.info("📅 **Período:** 2 anos de dados históricos")
            st.info("⏰ **Timeframe:** Diário (MT5)")
            bb_mt5_timeframe = mt5.TIMEFRAME_D1
        else:
            st.info("📅 **Período fixo:** 2 anos de dados históricos")
            st.info("⏰ **Timeframe fixo:** 1 dia")
            bb_mt5_timeframe = None

        # Fixed interval: 1 day
        interval_bb = "1d"

        st.markdown('</div>', unsafe_allow_html=True)

    # Analysis button
    analyze_button_bb = st.button("🚀 INICIAR DETECÇÃO DE TOPOS E FUNDOS", type="primary", use_container_width=True, key="analyze_bb")

    # Analysis logic for Bollinger Bands
    if analyze_button_bb:
        if not symbols_list_bb:
            st.error("Por favor selecione pelo menos um ativo para análise.")
            st.stop()

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            bb_results = []
            total_symbols = len(symbols_list_bb)

            for idx, current_symbol in enumerate(symbols_list_bb):
                status_text.text(f"Analisando {current_symbol} ({idx+1}/{total_symbols})...")
                progress_bar.progress(int((idx / total_symbols) * 100))

                try:
                    # Convert dates to strings
                    start_str = start_date_bb.strftime("%Y-%m-%d")
                    end_str = end_date_bb.strftime("%Y-%m-%d")

                    # Download data
                    df_temp = get_market_data(
                        current_symbol, 
                        start_str, 
                        end_str, 
                        interval_bb,
                        data_source=bb_data_source_key,
                        mt5_timeframe=bb_mt5_timeframe,
                        mt5_mode="range",
                        n_candles=1000
                    )

                    if df_temp is None or df_temp.empty:
                        bb_results.append({
                            'symbol': current_symbol,
                            'status': 'Erro - Sem dados',
                            'signal': 'N/A',
                            'current_price': 'N/A',
                            'banda_superior': 'N/A',
                            'banda_inferior': 'N/A',
                            'sma': 'N/A',
                            'distance_pct': 'N/A'
                        })
                        continue

                    # Calculate Bollinger Bands with fixed parameters
                    bb_period = 20
                    bb_std = 2.0
                    min_distance_pct = 0.0
                    
                    sma = df_temp['close'].rolling(window=bb_period).mean()
                    std = df_temp['close'].rolling(window=bb_period).std()
                    banda_superior = sma + (bb_std * std)
                    banda_inferior = sma - (bb_std * std)

                    # Get current values
                    current_price = df_temp['close'].iloc[-1]
                    current_banda_superior = banda_superior.iloc[-1]
                    current_banda_inferior = banda_inferior.iloc[-1]
                    current_sma = sma.iloc[-1]

                    # Determine signal
                    signal = 'Neutro'
                    distance_pct = 0

                    # Check if price is below lower band (potential bottom/buy signal)
                    if current_price < current_banda_inferior:
                        distance_pct = ((current_banda_inferior - current_price) / current_price) * 100
                        signal = 'Possível Fundo (Compra)'

                    # Check if price is above upper band (potential top/sell signal)
                    elif current_price > current_banda_superior:
                        distance_pct = ((current_price - current_banda_superior) / current_price) * 100
                        signal = 'Possível Topo (Venda)'

                    bb_results.append({
                        'symbol': current_symbol,
                        'status': 'Sucesso',
                        'signal': signal,
                        'current_price': current_price,
                        'banda_superior': current_banda_superior,
                        'banda_inferior': current_banda_inferior,
                        'sma': current_sma,
                        'distance_pct': distance_pct
                    })

                except Exception as e:
                    bb_results.append({
                        'symbol': current_symbol,
                        'status': f'Erro: {str(e)[:50]}...',
                        'signal': 'N/A',
                        'current_price': 'N/A',
                        'banda_superior': 'N/A',
                        'banda_inferior': 'N/A',
                        'sma': 'N/A',
                        'distance_pct': 'N/A'
                    })

            progress_bar.progress(100)
            status_text.text("Detecção Completa!")

            # Display results
            st.success(f"✅ Análise de Topos e Fundos completa para {len(symbols_list_bb)} ativos")

            # Use all results
            signal_results = bb_results
            # Display buying opportunities (potential bottoms) with a note on distance
            buy_opportunities = [r for r in signal_results if 'Compra' in r['signal']]
            if buy_opportunities:
                st.subheader(f"🟢 {len(buy_opportunities)} Oportunidade(s) de Compra Detectada(s)")

                for result in buy_opportunities:
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

                    with col1:
                        st.write(f"**{result['symbol']}**")
                    with col2:
                        st.write(f"Preço: {result['current_price']:.2f}")
                    with col3:
                        st.write(f"Distância: {result['distance_pct']:.2f}%")
                    with col4:
                        col4.empty()
                    with col5:
                        st.success("🟢 COMPRA")

                    st.markdown("---")

                st.info("ℹ️ Nota: Quanto maior a distância do ativo, maior a possibilidade de reversão.")

            # Display selling opportunities (potential tops) with a note on distance
            sell_opportunities = [r for r in signal_results if 'Venda' in r['signal']]
            if sell_opportunities:
                st.subheader(f"🔴 {len(sell_opportunities)} Oportunidade(s) de Venda Detectada(s)")

                for result in sell_opportunities:
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

                    with col1:
                        st.write(f"**{result['symbol']}**")
                    with col2:
                        st.write(f"Preço: {result['current_price']:.2f}")
                    with col3:
                        st.write(f"Distância: {result['distance_pct']:.2f}%")
                    with col4:
                        col4.empty()
                    with col5:
                        st.error("🔴 VENDA")

                    st.markdown("---")

                st.info("ℹ️ Nota: Quanto maior a distância do ativo, maior a possibilidade de reversão.")

            if not buy_opportunities and not sell_opportunities:
                st.info("ℹ️ Nenhuma oportunidade de compra ou venda detectada no período analisado.")

            # Summary metrics
            st.subheader("📊 Resumo da Análise")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_assets = len(bb_results)
                st.metric("Total de Ativos", total_assets)

            with col2:
                successful_analysis = len([r for r in bb_results if r['status'] == 'Sucesso'])
                st.metric("Análises Bem-sucedidas", successful_analysis)

            with col3:
                st.metric("Oportunidades de Compra", len(buy_opportunities))

            with col4:
                st.metric("Oportunidades de Venda", len(sell_opportunities))

            # Full results table
            st.subheader("📋 Resultados Detalhados")
            
            # Create summary dataframe with only essential columns
            summary_df = pd.DataFrame(bb_results)
            
            # Select only required columns
            essential_columns = ['symbol', 'status', 'signal']
            summary_df_display = summary_df[essential_columns].copy()
            
            # Rename columns for better display
            display_columns = {
                'symbol': 'Ativo',
                'status': 'Status',
                'signal': 'Sinal'
            }
            
            summary_df_display = summary_df_display.rename(columns=display_columns)
            st.dataframe(summary_df_display, use_container_width=True)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"Erro durante a análise: {str(e)}")
            st.write("Por favor verifique os parâmetros e tente novamente.")

with tab6:
    # Telegram Bot tab
    st.markdown("## 🤖 Bot de Alertas do Telegram")
    st.markdown("Manual de Instruções e Informações do Bot")

    # Bot information section
    st.markdown("### 📱 Informações do Bot")
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin: 0; color: #1f77b4;">🤖 Bot do Telegram: @Ovecchia_bot</h4>
        <p><strong>Funcionalidades:</strong></p>
        <ul>
            <li>🔍 Screening automático de múltiplos ativos</li>
            <li>📊 Detecção de topos e fundos</li>
            <li>⚡ Alertas em tempo real de mudanças de estado</li>
            <li>📈 Análise baseada em timeframe de 1 dia</li>
            <li>🎯 Estratégias: Agressiva, Balanceada e Conservadora</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # How to use section
    st.markdown("### 📋 Como Usar o Bot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Primeiros Passos")
        st.markdown("""
        **1. Adicione o bot:**
        No Telegram, procure por **@Ovecchia_bot** e clique em "Iniciar"
        
        **2. Comandos disponíveis:**
        - `/start` - Iniciar o bot e ver boas-vindas
        - `/analise [estrategia] [ativo] [timeframe] [data_inicio] [data_fim]` - Análise individual com gráfico
        - `/screening [estrategia] [ativos]` - Screening de múltiplos ativos
        - `/topos_fundos [ativos]` - Detectar topos e fundos
        - `/status` - Ver status do bot
        - `/help` - Ajuda detalhada com comandos
        """)

    with col2:
        st.markdown("#### ⚙️ Configurações")
        st.markdown("""
        **Estratégias disponíveis:**
        - **🔥 agressiva:** Mais sinais, maior frequência
        - **⚖️ balanceada:** Equilíbrio entre sinais e confiabilidade
        - **🛡️ conservadora:** Sinais mais confiáveis, menor frequência
        
        **Timeframes suportados:** 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk
        **Período padrão de análise:** Baseado no timeframe escolhido
        **Datas personalizadas:** Formato YYYY-MM-DD (opcional)
        """)

    # Bot status section
    st.markdown("### 📊 Informações do Bot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #1f77b4;">Status</h4>
            <h2 style="margin: 0; color: #333;">🟢 Online 24/7</h2>
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Bot está sempre ativo</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #1f77b4;">Bot Username</h4>
            <h2 style="margin: 0; color: #333;">@Ovecchia_bot</h2>
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Procure no Telegram</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #1f77b4;">Timeframe</h4>
            <h2 style="margin: 0; color: #333;">Flexível</h2>
            <p style="margin: 0; font-size: 0.9rem; color: #666;">Comandos screening e topos e fundos são exclusivo para 1d . Análise do ativo pode ser personalizada.</p>
        </div>
        """, unsafe_allow_html=True)

    # Instructions to use the bot
    st.markdown("### 🚀 Como Começar")
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #1f77b4;">Passos para usar o bot:</h4>
        <ol style="color: #333;">
            <li><strong>Abra o Telegram</strong> no seu celular ou computador</li>
            <li><strong>Procure por:</strong> <code>@Ovecchia_bot</code></li>
            <li><strong>Clique em "Iniciar"</strong> ou digite <code>/start</code></li>
            <li><strong>Pronto!</strong> O bot responderá com as opções disponíveis</li>
        </ol>
        <p style="margin-top: 1rem;"><strong>💡 Exemplos de comandos:</strong></p>
        <ul style="color: #333;">
            <li><code>/analise balanceada PETR4.SA 1d</code> - Análise da Petrobras</li>
            <li><code>/screening balanceada BTC-USD ETH-USD</code> - Screening de criptos</li>
            <li><code>/topos_fundos PETR4.SA VALE3.SA</code> - Detectar extremos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Example alerts section
    st.markdown("### 📢 Exemplos de Alertas")
    
    st.markdown("""
    <div class="metric-card">
        <p><strong>🔍 Exemplo de Screening:</strong></p>
        <div style="background: #f0f2f6; padding: 0.75rem; border-radius: 8px; font-family: monospace;">
            🚨 ALERTAS DE MUDANÇA DE ESTADO<br><br>
            📊 Estratégia: Balanceado<br>
            ⏰ Timeframe: 1 dia<br><br>
            🟢 BTC-USD<br>
            💰 Preço: 45,230.50<br>
            📈 ⚫ Stay Out → 🟢 Buy<br><br>
            🔴 ETH-USD<br>
            💰 Preço: 2,850.75<br>
            📈 🟢 Buy → 🔴 Sell
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <p><strong>📊 Exemplo de Análise Individual:</strong></p>
        <div style="background: #f0f2f6; padding: 0.75rem; border-radius: 8px; font-family: monospace;">
            📊 OVECCHIA TRADING - PETR4.SA<br>
            🎯 Balanceado | ⏰ 1D<br>
            📅 Período: 2024-01-01 até 2024-12-01<br><br>
            [Gráfico de análise enviado como imagem]
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <p><strong>📊 Exemplo de Topos e Fundos:</strong></p>
        <div style="background: #f0f2f6; padding: 0.75rem; border-radius: 8px; font-family: monospace;">
            📊 DETECÇÃO DE TOPOS E FUNDOS<br>
            ⏰ Timeframe: 1 dia<br><br>
            🟢 POSSÍVEL FUNDO (COMPRA):<br>
            • PETR4.SA: 28.45<br>
            📊 Distância: 2.30%<br><br>
            🔴 POSSÍVEL TOPO (VENDA):<br>
            • VALE3.SA: 72.80<br>
            📊 Distância: 1.80%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical information
    st.markdown("### 🔧 Informações Técnicas")
    
    st.markdown("""
    <div class="metric-card">
        <p><strong>Especificações do Bot:</strong></p>
        <ul>
            <li><strong>Polling:</strong> Verifica mensagens a cada 2 segundos</li>
            <li><strong>Timeout:</strong> 10 segundos para requisições</li>
            <li><strong>Análise automática:</strong> A cada 4 horas (configurável)</li>
            <li><strong>Fonte de dados:</strong> Yahoo Finance API</li>
            <li><strong>Período de dados:</strong> 365 dias históricos</li>
            <li><strong>Processamento:</strong> Thread separada para não bloquear interface</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab7:
    # About tab
    st.markdown("## ℹ️ Sobre o Sistema OVECCHIA TRADING")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Missão")
        st.markdown("""
        <div class="metric-card">
            <p>O Sistema OVECCHIA TRADING foi desenvolvido para democratizar o acesso a análises técnicas avançadas, 
            oferecendo ferramentas profissionais de trading quantitativo de forma acessível e intuitiva.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔬 Metodologia")
        st.markdown("""
        <div class="metric-card">
            <p><strong>Sistema avançado que combina múltiplos indicadores técnicos e financeiros com inteligência artificial:</strong></p>
            <p>Utiliza-se de técnicas modernas para identificar oportunidades de negociação, determinar pontos de entrada e saída mais eficientes, assegurando uma gestão de risco sofisticada e adaptada às condições de mercado.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 Recursos Disponíveis")
        st.markdown("""
        <div class="metric-card">
            <ul>
                <li><strong>Análise Individual:</strong> Estudo detalhado de um ativo</li>
                <li><strong>Screening Multi-Ativos:</strong> Monitoramento de carteiras</li>
                <li><strong>Otimização Automática:</strong> Busca pelos melhores parâmetros</li>
                <li><strong>Múltiplos Timeframes:</strong> De 1 minuto a 3 meses</li>
                <li><strong>Critérios de Saída:</strong> Stop Loss, Alvo Fixo, Tempo, MM</li>
                <li><strong>Direções de Trading:</strong> Long, Short ou Ambos</li>
                <li><strong>Detecção de Topos e Fundos:</strong> Identificação de reversões potenciais</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛡️ Gerenciamento de Risco")
        st.markdown("""
        <div class="metric-card">
            <p><strong>Sistema de Stop Loss Baseado em Métricas Matemáticas:</strong></p>
            <p>Oferecemos diferentes níveis de stop para atender a diversos perfis de investidores:</p>
            <ul>
                <li><strong>Stop Justo:</strong> para investidores mais conservadores</li>
                <li><strong>Stop Balanceado:</strong> uma abordagem equilibrada</li>
                <li><strong>Stop Largo:</strong> para investidores mais agressivos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Ativos Suportados")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 🪙 Criptomoedas")
        st.markdown("- Bitcoin (BTC-USD)")
        st.markdown("- Ethereum (ETH-USD)")
        st.markdown("- Exemplos: Binance Coin (BNB-USD), Cardano (ADA-USD)")
        st.markdown("- Todos os listados no Yahoo Finance")

    with col2:
        st.markdown("#### 🇧🇷 Ações Brasileiras")
        st.markdown("- Petrobras (PETR4.SA)")
        st.markdown("- Vale (VALE3.SA)")
        st.markdown("- Exemplos: Itaú Unibanco (ITUB4.SA), Bradesco (BBDC4.SA)")
        st.markdown("- Todas as listadas no Yahoo Finance")

    with col3:
        st.markdown("#### 🇺🇸 Ações Americanas")
        st.markdown("- Apple (AAPL)")
        st.markdown("- Microsoft (MSFT)")
        st.markdown("- Exemplos: Google (GOOGL), Amazon (AMZN)")
        st.markdown("- Todas as listadas no Yahoo Finance")

    with col4:
        st.markdown("#### 💱 Forex & Commodities")
        st.markdown("- EUR/USD")
        st.markdown("- Ouro (GC=F)")
        st.markdown("- Exemplos: GBP/USD, Petróleo bruto (CL=F)")
        st.markdown("- Todos os listados no Yahoo Finance")

    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
 <div style="background: linear-gradient(90deg, #e3f2fd, #f3e5f5); padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; color: black;">
        <p><strong>⚠️ AVISO IMPORTANTE:</strong></p>
        <p>Este sistema é desenvolvido para fins educacionais e de pesquisa. As análises e sinais gerados 
        <strong>NÃO constituem recomendações de investimento</strong>. Trading e investimentos envolvem riscos 
        significativos e você pode perder parte ou todo o seu capital investido.</p>
        <p><strong>Sempre consulte um profissional qualificado antes de tomar decisões de investimento.</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Atualizações Recentes do Bot Telegram")
    st.markdown(
        """
        <div class=\"metric-card\" style=\"border-left: 4px solid #25D366;\">
            <p><strong>🚀 Versão 2.0 - Bot @Ovecchia_bot</strong></p>
            <ul>
                <li><strong>📊 Nova Funcionalidade:</strong> Análise Individual com Gráficos Interativos</li>
                <li><strong>🔄 Comando Aprimorado:</strong> Estrutura mais intuitiva e funcional</li>
                <li><strong>📅 Períodos Personalizados:</strong> Ajuste flexível das datas de análise</li>
                <li><strong>⌚ Múltiplos Timeframes:</strong> Variedade de intervalos de tempo, de 1 minuto a 1 semana</li>
                <li><strong>💾 Performance Otimizada:</strong> Processamento acelerado e eficiente dos dados</li>
                <li><strong>❗ Tratamento Avançado de Erros:</strong> Alertas mais informativos para melhor usabilidade</li>
                <li><strong>🔍 Validação Automática:</strong> Formatos de data são conferidos instantaneamente</li>
                <li><strong>🧹 Manutenção Automática:</strong> Gerenciamento automático de arquivos temporários</li>
            </ul>
            <div style=\"background: #f0f8f0; padding: 1rem; border-radius: 10px; margin-top: 1rem;\">
                <p style=\"color: #25D366;\"><strong>💡 Dica:</strong> Explore períodos personalizados para investigar eventos de mercado específicos!</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### 👨‍💻 Desenvolvido por")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #e3f2fd, #f3e5f5); padding: 1rem; border-radius: 10px; text-align: center;">
        <h3 style="color: #1976d2; margin: 0;">OVECCHIA TRADING</h3>
        <p style="margin: 0; color: #666;">Sistema Avançado de Análise Técnica Quantitativa</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; margin-top: 2rem;">
    <p style="color: #666; margin: 0;"><strong>OVECCHIA TRADING - MODELO QUANT</strong></p>
    <p style="color: #999; font-size: 0.9rem; margin: 0;">⚠️ Para fins educacionais apenas. Não constitui recomendação financeira.</p>
</div>
""", unsafe_allow_html=True)