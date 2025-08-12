#!/usr/bin/env python3
import telebot
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import threading
import time
import os
import sys
import difflib
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier
from binance.client import Client

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot token
BOT_TOKEN = "8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k"

# Initialize bot
bot = telebot.TeleBot(BOT_TOKEN)

# Funções auxiliares para tolerância a erros
def normalize_text(text):
    """Normaliza texto removendo acentos e convertendo para minúsculas"""
    if not text:
        return ""
    # Remove acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    # Converte para minúsculas
    return text.lower().strip()

def calculate_similarity(text1, text2):
    """Calcula similaridade entre dois textos usando SequenceMatcher"""
    return difflib.SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()

def find_best_match(input_text, options, threshold=0.6):
    """Encontra a melhor correspondência em uma lista de opções"""
    if not input_text or not options:
        return None
    
    normalized_input = normalize_text(input_text)
    best_match = None
    best_score = 0
    
    for option in options:
        score = calculate_similarity(normalized_input, option)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = option
    
    return best_match

def fuzzy_command_match(user_input):
    """Identifica comandos com tolerância a erros"""
    commands = {
        'start': ['start', 'iniciar', 'comecar', 'inicio'],
        'analise': ['analise', 'analisar', 'analysis', 'analyze', 'grafico', 'chart'],
        'screening': ['screening', 'screnning', 'screning', 'screen', 'varredura', 'busca'],
        'topos_fundos': ['topos_fundos', 'toposfundos', 'topo_fundo', 'topofundo', 'reversao', 'oportunidades'],
        'status': ['status', 'estado', 'situacao', 'info'],
        'restart': ['restart', 'reiniciar', 'reboot', 'reset'],
        'help': ['help', 'ajuda', 'ajudar', 'comandos', '?']
    }
    
    user_input = normalize_text(user_input.replace('/', ''))
    
    for command, variations in commands.items():
        for variation in variations:
            if calculate_similarity(user_input, variation) >= 0.7:
                return command
    
    return None

def fuzzy_strategy_match(user_input):
    """Identifica estratégias com tolerância a erros"""
    strategies = {
        'agressiva': ['agressiva', 'agressivo', 'agressive', 'rapida', 'forte'],
        'balanceada': ['balanceada', 'balanceado', 'balanced', 'equilibrada', 'media', 'normal'],
        'conservadora': ['conservadora', 'conservador', 'conservative', 'segura', 'cautelosa']
    }
    
    normalized_input = normalize_text(user_input)
    
    for strategy, variations in strategies.items():
        for variation in variations:
            if calculate_similarity(normalized_input, variation) >= 0.7:
                return strategy
    
    return None

def fuzzy_list_match(user_input):
    """Identifica listas com tolerância a erros"""
    lists = {
        'açõesbr': ['acoesbr', 'açõesbr', 'acoes_br', 'açoes_br', 'brasileiras', 'brasil', 'br'],
        'açõeseua': ['acoeseua', 'açõeseua', 'acoes_eua', 'açoes_eua', 'americanas', 'eua', 'usa', 'us'],
        'criptos': ['criptos', 'crypto', 'cripto', 'moedas', 'bitcoin', 'criptomoedas'],
        'forex': ['forex', 'fx', 'cambio', 'moedas', 'divisas'],
        'commodities': ['commodities', 'commodity', 'mercadorias', 'materias']
    }
    
    normalized_input = normalize_text(user_input)
    
    for list_name, variations in lists.items():
        for variation in variations:
            if calculate_similarity(normalized_input, variation) >= 0.7:
                return list_name
    
    return None

def parse_flexible_command(message_text):
    """Analisa comandos com tolerância a erros"""
    parts = message_text.strip().split()
    if not parts:
        return None
    
    # Identificar comando
    first_part = parts[0]
    if first_part.startswith('/'):
        command = fuzzy_command_match(first_part)
    else:
        command = fuzzy_command_match(first_part)
    
    if not command:
        return None
    
    # Processar argumentos baseado no comando
    args = parts[1:] if len(parts) > 1 else []
    processed_args = []
    
    for arg in args:
        # Tentar identificar estratégia
        strategy = fuzzy_strategy_match(arg)
        if strategy:
            processed_args.append(strategy)
            continue
        
        # Tentar identificar lista
        list_match = fuzzy_list_match(arg)
        if list_match:
            processed_args.append(list_match)
            continue
        
        # Manter argumento original se não encontrar correspondência
        processed_args.append(arg)
    
    return {
        'command': command,
        'args': processed_args,
        'original_text': message_text
    }

class OvecchiaTradingBot:
    def __init__(self):
        self.users_config = {}

    def get_binance_data(self, symbol, start_date, end_date, interval="1d"):
        """Função para coletar dados da Binance API"""
        try:
            # Credenciais da Binance
            api_key = 'kagmRJ2TNwr7No68i3aRG2MZdm5MPDTBncwnrKUv2Wv8arpXXxuikVUij981Wxvu'
            api_secret = 'VAYi1m9r0sLy7T8vbqPBSxaOjL4BI58KnMS13USRatbULqrUdoDJnILjyyz4skgx'
            
            # Inicializa o cliente
            client = Client(api_key, api_secret)
            
            # Mapear intervalos
            interval_mapping = {
                "1m": Client.KLINE_INTERVAL_1MINUTE,
                "5m": Client.KLINE_INTERVAL_5MINUTE,
                "15m": Client.KLINE_INTERVAL_15MINUTE,
                "30m": Client.KLINE_INTERVAL_30MINUTE,
                "1h": Client.KLINE_INTERVAL_1HOUR,
                "4h": Client.KLINE_INTERVAL_4HOUR,
                "1d": Client.KLINE_INTERVAL_1DAY,
                "1wk": Client.KLINE_INTERVAL_1WEEK
            }
            
            binance_interval = interval_mapping.get(interval, Client.KLINE_INTERVAL_1DAY)
            
            # Converter símbolo para formato Binance
            binance_symbol = symbol.replace('-USD', 'USDT').replace('-USDT', 'USDT')
            
            # Converter datas
            if isinstance(start_date, str):
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                start_datetime = start_date
                
            if isinstance(end_date, str):
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_datetime = end_date
            
            # Puxar dados históricos
            klines = client.get_historical_klines(
                binance_symbol, 
                binance_interval, 
                start_datetime.strftime("%d %b %Y"),
                end_datetime.strftime("%d %b %Y")
            )
            
            if not klines:
                return pd.DataFrame()
            
            # Criar DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Selecionar apenas colunas necessárias
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Converter timestamp para datetime
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)
            
            # Converter para float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Reordenar colunas
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Dados Binance coletados com sucesso para {symbol}: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Erro ao coletar dados Binance para {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_market_data(self, symbol, start_date, end_date, interval="1d", data_source="yahoo"):
        """Função para coletar dados do mercado"""
        try:
            logger.info(f"Coletando dados para {symbol} via {data_source}")
            
            # Detectar automaticamente se é cripto e usar Binance se solicitado
            is_crypto = any(symbol.upper().endswith(suffix) for suffix in ['USDT', '-USD', 'USD'])
            
            if data_source == "binance" and is_crypto:
                return self.get_binance_data(symbol, start_date, end_date, interval)
            else:
                # Yahoo Finance (código original)
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)

                if df is None or df.empty:
                    logger.warning(f"Sem dados para {symbol}")
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

                logger.info(f"Dados coletados com sucesso para {symbol}: {len(df)} registros")
                return df
                
        except Exception as e:
            logger.error(f"Erro ao coletar dados para {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_ovelha_v2_signals(self, df, strategy_type="Balanceado", sma_short=60, sma_long=70, lookahead=3, threshold=0.002, buffer=0.0015):
        """Função para calcular sinais usando o modelo OVELHA V2 com Random Forest"""
        try:
            if df.empty:
                return df

            # Definir parâmetros baseado na estratégia
            if strategy_type == "Agressivo":
                sma_short = 10
                sma_long = 21
            elif strategy_type == "Conservador":
                sma_short = 140
                sma_long = 200
            else:  # Balanceado
                sma_short = 60
                sma_long = 70

            # =======================
            # CÁLCULO DAS FEATURES
            # =======================
            # SMAs
            df[f'SMA_{sma_short}'] = df['close'].rolling(window=sma_short).mean()
            df[f'SMA_{sma_long}'] = df['close'].rolling(window=sma_long).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()

            # RSI(14)
            delta = df['close'].diff()
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
            avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['RSI_14'] = 100 - (100 / (1 + rs))
            df['RSI_14'] = df['RSI_14'].fillna(method='bfill')

            # RSL(20)
            df['RSL_20'] = df['close'] / df['SMA_20']

            # ATR(14)
            df['prior_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = (df['high'] - df['prior_close']).abs()
            df['tr3'] = (df['low'] - df['prior_close']).abs()
            df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # Retorno, aceleração e volatilidade intrínseca
            df['ret_1'] = df['close'].pct_change()
            df['accel'] = df['ret_1'].diff()
            df['decel'] = -df['accel']
            df['atr_norm'] = df['ATR'] / df['close']

            # =======================
            # CRIAÇÃO DO TARGET Y
            # =======================
            df['future_ret'] = df['close'].shift(-lookahead) / df['close'] - 1
            df['y'] = 0
            df.loc[df['future_ret'] > threshold, 'y'] = 1
            df.loc[df['future_ret'] < -threshold, 'y'] = -1

            # =======================
            # TREINAMENTO DO MODELO
            # =======================
            features = ['RSI_14', 'RSL_20', 'ATR', 'ret_1', 'accel', 'decel', 'atr_norm']
            X = df[features].dropna()
            y = df.loc[X.index, 'y']

            # Verificar se temos dados suficientes para treinar
            if len(X) < 50:
                logger.warning("Dados insuficientes para OVELHA V2, usando modelo clássico")
                return None

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X, y)

            # =======================
            # PREVISÕES
            # =======================
            df['Signal_model'] = np.nan
            df.loc[X.index, 'Signal_model'] = model.predict(X)

            # =======================
            # FILTRO DE TENDÊNCIA + HISTERESE
            # =======================
            df['Signal'] = 'Stay Out'
            for i in range(1, len(df)):
                prev_estado = df['Signal'].iloc[i-1]

                price = df['close'].iloc[i]
                sma_s = df[f'SMA_{sma_short}'].iloc[i]
                sma_l = df[f'SMA_{sma_long}'].iloc[i]

                # BUY - com buffer para evitar falsos cruzamentos
                if df['Signal_model'].iloc[i] == 1:
                    if price > sma_s * (1 + buffer) and price > sma_l * (1 + buffer):
                        df.loc[df.index[i], 'Signal'] = 'Buy'
                    else:
                        df.loc[df.index[i], 'Signal'] = prev_estado

                # SELL - com buffer para evitar falsos cruzamentos
                elif df['Signal_model'].iloc[i] == -1:
                    if price < sma_s * (1 - buffer):
                        df.loc[df.index[i], 'Signal'] = 'Sell'
                    else:
                        df.loc[df.index[i], 'Signal'] = prev_estado
                else:
                    df.loc[df.index[i], 'Signal'] = prev_estado

            # State persistence
            df['Estado'] = 'Stay Out'
            for i in range(len(df)):
                if i == 0:
                    continue

                estado_anterior = df['Estado'].iloc[i - 1]
                sinal_atual = df['Signal'].iloc[i]

                if sinal_atual != 'Stay Out':
                    df.loc[df.index[i], 'Estado'] = sinal_atual
                else:
                    df.loc[df.index[i], 'Estado'] = estado_anterior

            return df

        except Exception as e:
            logger.error(f"Erro no modelo OVELHA V2: {str(e)}")
            return None

    def calculate_indicators_and_signals(self, df, strategy_type="Balanceado"):
        """Calcula indicadores e gera sinais"""
        if df.empty:
            return df

        try:
            # Definir parâmetros baseado na estratégia
            if strategy_type == "Agressivo":
                sma_short = 10
                sma_long = 21
            elif strategy_type == "Conservador":
                sma_short = 140
                sma_long = 200
            else:  # Balanceado
                sma_short = 60
                sma_long = 70

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
                except Exception as e:
                    logger.error(f"Erro no cálculo de sinais na linha {i}: {str(e)}")
                    continue

            # State persistence
            df['Estado'] = 'Stay Out'
            for i in range(len(df)):
                if i == 0:
                    continue

                estado_anterior = df['Estado'].iloc[i - 1]
                sinal_atual = df['Signal'].iloc[i]

                if sinal_atual != 'Stay Out':
                    df.loc[df.index[i], 'Estado'] = sinal_atual
                else:
                    df.loc[df.index[i], 'Estado'] = estado_anterior

            return df
        except Exception as e:
            logger.error(f"Erro no cálculo de indicadores: {str(e)}")
            return df

    def perform_screening(self, symbols_list, strategy_type="Balanceado"):
        """Realiza screening de múltiplos ativos"""
        results = []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=730)  # 2 years

        for symbol in symbols_list:
            try:
                logger.info(f"Analisando {symbol}")
                df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"), "1d")

                if df.empty:
                    logger.warning(f"Sem dados para {symbol}")
                    continue

                df = self.calculate_indicators_and_signals(df, strategy_type)

                if len(df) > 1:
                    current_state = df['Estado'].iloc[-1]
                    previous_state = df['Estado'].iloc[-2]

                    if current_state != previous_state:
                        results.append({
                            'symbol': symbol,
                            'current_state': current_state,
                            'previous_state': previous_state,
                            'current_price': df['close'].iloc[-1]
                        })

            except Exception as e:
                logger.error(f"Erro ao analisar {symbol}: {str(e)}")
                continue

        return results

    def detect_tops_bottoms(self, symbols_list):
        """Detecta topos e fundos usando Bollinger Bands"""
        results = []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=730)  # 2 years

        for symbol in symbols_list:
            try:
                df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"), "1d")

                if df.empty:
                    continue

                # Calculate Bollinger Bands
                bb_period = 20
                bb_std = 2.0

                sma = df['close'].rolling(window=bb_period).mean()
                std = df['close'].rolling(window=bb_period).std()
                banda_superior = sma + (bb_std * std)
                banda_inferior = sma - (bb_std * std)

                current_price = df['close'].iloc[-1]
                current_banda_superior = banda_superior.iloc[-1]
                current_banda_inferior = banda_inferior.iloc[-1]

                signal = None
                distance_pct = 0

                if current_price < current_banda_inferior:
                    distance_pct = ((current_banda_inferior - current_price) / current_price) * 100
                    signal = 'Possível Fundo (Compra)'
                elif current_price > current_banda_superior:
                    distance_pct = ((current_price - current_banda_superior) / current_price) * 100
                    signal = 'Possível Topo (Venda)'

                if signal:
                    results.append({
                        'symbol': symbol,
                        'signal': signal,
                        'current_price': current_price,
                        'distance_pct': distance_pct
                    })

            except Exception as e:
                logger.error(f"Erro ao analisar topos/fundos {symbol}: {str(e)}")
                continue

        return results

    def generate_analysis_chart(self, symbol, strategy_type, timeframe, model_type="ovelha", custom_start_date=None, custom_end_date=None):
        """Gera gráfico de análise para um ativo específico usando matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle
            import tempfile
            import os

            # Define período baseado no timeframe ou usa datas personalizadas
            if custom_start_date and custom_end_date:
                start_date = datetime.strptime(custom_start_date, '%Y-%m-%d').date()
                end_date = datetime.strptime(custom_end_date, '%Y-%m-%d').date()
            else:
                if timeframe in ['1m', '5m', '15m', '30m']:
                    days = 7  # 1 semana para timeframes menores
                elif timeframe in ['1h', '4h']:
                    days = 30  # 1 mês para timeframes de horas
                else:
                    days = 180  # 6 meses para timeframes maiores

                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days)

            # Coletar dados
            df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"), 
                                    end_date.strftime("%Y-%m-%d"), timeframe)

            if df.empty:
                return {'success': False, 'error': f'Sem dados encontrados para {symbol}'}

            # Calcular indicadores e sinais baseado no modelo escolhido
            if model_type == "ovelha2":
                df_v2 = self.calculate_ovelha_v2_signals(df, strategy_type)
                if df_v2 is not None:
                    df = df_v2
                    model_used = "OVELHA V2"
                else:
                    df = self.calculate_indicators_and_signals(df, strategy_type)
                    model_used = "OVELHA (fallback)"
            else:
                df = self.calculate_indicators_and_signals(df, strategy_type)
                model_used = "OVELHA"

            if df.empty:
                return {'success': False, 'error': 'Erro ao calcular indicadores'}

            # Preparar dados para matplotlib
            df['time'] = pd.to_datetime(df['time'])

            # Color coding
            df['Color'] = 'black'
            df.loc[df['Estado'] == 'Buy', 'Color'] = 'blue'
            df.loc[df['Estado'] == 'Sell', 'Color'] = 'red'

            # Create indicator mapping
            estado_mapping = {'Buy': 1, 'Sell': 0, 'Stay Out': 0.5}
            df['Indicator'] = df['Estado'].apply(lambda x: estado_mapping.get(x, 0.5))

            # Criar figura com subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                         gridspec_kw={'height_ratios': [3, 1]}, 
                                         sharex=True)

            # Título principal
            titulo_grafico = f"OVECCHIA TRADING - {symbol} - {model_used} - {timeframe.upper()}"
            fig.suptitle(titulo_grafico, fontsize=16, fontweight='bold')

            # Subplot 1: Preço com sinais
            ax1.set_title("Gráfico do Preço com Sinais", fontsize=12)

            # Plotar linha de preço com cores baseadas no estado
            for i in range(len(df) - 1):
                color = df['Color'].iloc[i]
                ax1.plot(df['time'].iloc[i:i+2], df['close'].iloc[i:i+2], 
                        color=color, linewidth=2)

            ax1.set_ylabel('Preço', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Adicionar legenda
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, label='Sinal de Compra'),
                Line2D([0], [0], color='red', lw=2, label='Sinal de Venda'),
                Line2D([0], [0], color='black', lw=2, label='Ficar de Fora')
            ]
            ax1.legend(handles=legend_elements, loc='upper left')

            # Subplot 2: Indicador de sinais
            ax2.set_title("Indicador de Sinais", fontsize=12)
            ax2.plot(df['time'], df['Indicator'], color='purple', linewidth=2, marker='o', markersize=2)
            ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
            ax2.set_ylabel('Sinal', fontsize=10)
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_yticks([0, 0.5, 1])
            ax2.set_yticklabels(['Venda', 'Ficar de Fora', 'Compra'])
            ax2.grid(True, alpha=0.3)

            # Formatação do eixo X
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            # Ajustar layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            # Salvar gráfico
            temp_dir = tempfile.gettempdir()
            chart_filename = f"chart_{symbol.replace('.', '_').replace('-', '_')}_{int(datetime.now().timestamp())}.png"
            chart_path = os.path.join(temp_dir, chart_filename)

            plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()  # Fechar figura para liberar memória

            # Verificar se o arquivo foi criado
            if not os.path.exists(chart_path):
                return {'success': False, 'error': 'Falha ao gerar arquivo de imagem'}

            # Caption com informações completas
            if custom_start_date and custom_end_date:
                caption = f"📊 OVECCHIA TRADING - {symbol}\n🤖 {model_used} | 🎯 {strategy_type} | ⏰ {timeframe.upper()}\n📅 {custom_start_date} até {custom_end_date}"
            else:
                caption = f"📊 OVECCHIA TRADING - {symbol}\n🤖 {model_used} | 🎯 {strategy_type} | ⏰ {timeframe.upper()}\n📅 Período: {start_date} até {end_date}"

            return {
                'success': True,
                'chart_path': chart_path,
                'caption': caption
            }

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico para {symbol}: {str(e)}")
            return {'success': False, 'error': f'Erro ao gerar análise: {str(e)}'}

# Initialize bot instance
trading_bot = OvecchiaTradingBot()

# Command handlers
@bot.message_handler(commands=['start'])
def start_command(message):
    try:
        user_name = message.from_user.first_name
        user_id = message.from_user.id
        logger.info(f"Comando /start recebido de {user_name} (ID: {user_id})")

        welcome_message = """🤖 Bem-vindo ao OVECCHIA TRADING BOT!

👋 Olá! Sou o bot oficial do sistema OVECCHIA TRADING, desenvolvido para fornecer análises técnicas avançadas e sinais de trading profissionais.

📊 FUNCIONALIDADES PRINCIPAIS:
• Análise individual de ativos com gráficos
• Screening automático de múltiplos ativos
• Detecção de topos e fundos
• Alertas em tempo real de mudanças de estado
• Suporte a múltiplas estratégias de trading

🎯 COMANDOS DISPONÍVEIS:
/analise [estrategia] [ativo] [timeframe] - Análise completa com gráfico
/screening [estrategia] [ativos] - Screening de múltiplos ativos
/topos_fundos [ativos] - Detectar oportunidades de reversão
/status - Verificar status do bot
/help - Ajuda detalhada

📈 ESTRATÉGIAS:
• agressiva - Mais sinais, maior frequência
• balanceada - Equilíbrio ideal (recomendada)
• conservadora - Sinais mais confiáveis

⏰ TIMEFRAMES SUPORTADOS:
1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk

🚀 EXEMPLO DE USO:
/analise balanceada PETR4.SA 1d

Comece agora mesmo digitando um comando!"""

        bot.reply_to(message, welcome_message)
        logger.info(f"Mensagem de boas-vindas enviada para {user_name}")
    except Exception as e:
        logger.error(f"Erro no comando /start: {str(e)}")
        bot.reply_to(message, "❌ Erro interno. Tente novamente mais tarde.")

@bot.message_handler(commands=['screening'])
def screening_command(message):
    try:
        user_name = message.from_user.first_name
        logger.info(f"Comando /screening recebido de {user_name}")

        # Parse arguments with fuzzy matching
        parsed = parse_flexible_command(message.text)
        if parsed and parsed['command'] == 'screening':
            args = parsed['args']
        else:
            args = message.text.split()[1:]  # Fallback para método original

        # Listas pré-definidas
        predefined_lists = {
            'açõesbr': [
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
            'açõeseua': [
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
            'criptos': [
                "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD",
                "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD",
                "TRX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
                "FIL-USD", "APT-USD", "ARB-USD", "NEAR-USD", "VET-USD"
            ],
            'forex': ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X"],
            'commodities': ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "KE=F", "CC=F", "KC=F"]
        }

        if not args:
            help_message = """
🔍 *SCREENING DE ATIVOS*

📝 *Como usar:*
/screening [estrategia] [lista/ativos]

🎯 *Estratégias disponíveis:*
• agressiva - Mais sinais
• balanceada - Equilibrada (padrão)
• conservadora - Sinais mais confiáveis

📊 *Listas pré-definidas:*
• açõesBR - Ações brasileiras
• açõesEUA - Ações americanas
• criptos - Criptomoedas
• forex - Pares de moedas
• commodities - Commodities

⏰ *Configurações fixas:*
• Timeframe: 1 dia (fixo)
• Período: 2 anos de dados históricos

📈 *Exemplos:*
`/screening balanceada açõesBR`
`/screening agressiva açõesEUA`
`/screening conservadora criptos`
`/screening balanceada BTC-USD ETH-USD PETR4.SA`

💡 *Nota:* Você pode usar listas pré-definidas OU especificar ativos individuais
            """
            bot.reply_to(message, help_message, parse_mode='Markdown')
            return

        bot.reply_to(message, "🔄 Processando screening...", parse_mode='Markdown')

        strategy = "Balanceado"
        symbols = []

        # Verificar se o primeiro argumento é uma estratégia
        if args[0].lower() in ['agressiva', 'balanceada', 'conservadora']:
            strategy_map = {
                'agressiva': 'Agressivo',
                'balanceada': 'Balanceado',
                'conservadora': 'Conservador'
            }
            strategy = strategy_map[args[0].lower()]
            remaining_args = args[1:]
        else:
            remaining_args = args

        # Verificar se é uma lista pré-definida ou ativos individuais
        if len(remaining_args) == 1 and remaining_args[0].lower() in predefined_lists:
            list_name = remaining_args[0].lower()
            symbols = predefined_lists[list_name]
            list_display_name = {
                'açõesbr': 'Ações Brasileiras',
                'açõeseua': 'Ações Americanas',
                'criptos': 'Criptomoedas',
                'forex': 'Forex',
                'commodities': 'Commodities'
            }
            bot.reply_to(message, f"📊 Analisando lista: {list_display_name[list_name]} ({len(symbols)} ativos)", parse_mode='Markdown')
        else:
            symbols = remaining_args

        if not symbols:
            bot.reply_to(message, "❌ Por favor, forneça uma lista válida ou pelo menos um ativo para análise.", parse_mode='Markdown')
            return

        logger.info(f"Realizando screening para {len(symbols)} ativos com estratégia {strategy}")

        # Realizar screening (limitado a 50 ativos por vez para evitar timeout)
        if len(symbols) > 50:
            bot.reply_to(message, f"⚠️ Lista muito grande ({len(symbols)} ativos). Analisando os ativos...", parse_mode='Markdown')
            symbols = symbols[:200]

        # Realizar screening
        results = trading_bot.perform_screening(symbols, strategy)

        if results:
            # Data atual da análise
            data_analise = datetime.now().strftime("%d/%m/%Y")
            
            response = f"🚨 *ALERTAS DE MUDANÇA DE ESTADO*\n📅 {data_analise}\n\n📊 Estratégia: {strategy}\n⏰ Timeframe: 1 dia (fixo)\n📅 Período: 2 anos de dados\n📈 Total analisado: {len(symbols)} ativos\n\n"

            for result in results:
                state_icon = "🟢" if result['current_state'] == "Buy" else "🔴" if result['current_state'] == "Sell" else "⚫"
                prev_icon = "🟢" if result['previous_state'] == "Buy" else "🔴" if result['previous_state'] == "Sell" else "⚫"

                response += f"{state_icon} *{result['symbol']}*\n"
                response += f"💰 Preço: {result['current_price']:.2f}\n"
                response += f"📈 {prev_icon} {result['previous_state']} → {state_icon} {result['current_state']}\n\n"

            # Dividir mensagem se muito longa
            if len(response) > 4000:
                parts = response.split('\n\n')
                current_message = f"🚨 *ALERTAS DE MUDANÇA DE ESTADO*\n📅 {data_analise}\n\n📊 Estratégia: {strategy}\n⏰ Timeframe: 1 dia\n📈 Total analisado: {len(symbols)} ativos\n\n"
                
                for part in parts[1:]:  # Skip header
                    if len(current_message + part + '\n\n') > 4000:
                        bot.reply_to(message, current_message, parse_mode='Markdown')
                        current_message = part + '\n\n'
                    else:
                        current_message += part + '\n\n'
                
                if current_message.strip():
                    bot.reply_to(message, current_message, parse_mode='Markdown')
            else:
                bot.reply_to(message, response, parse_mode='Markdown')

            logger.info(f"Screening enviado para {user_name}: {len(results)} alertas de {len(symbols)} ativos")
        else:
            bot.reply_to(message, f"ℹ️ Nenhuma mudança de estado detectada nos {len(symbols)} ativos analisados.", parse_mode='Markdown')
            logger.info(f"Nenhum alerta encontrado para {user_name}")

    except Exception as e:
        logger.error(f"Erro no comando /screening: {str(e)}")
        bot.reply_to(message, "❌ Erro ao processar screening. Tente novamente.")

@bot.message_handler(commands=['topos_fundos'])
def topos_fundos_command(message):
    try:
        user_name = message.from_user.first_name
        logger.info(f"Comando /topos_fundos recebido de {user_name}")

        # Parse arguments with fuzzy matching
        parsed = parse_flexible_command(message.text)
        if parsed and parsed['command'] == 'topos_fundos':
            args = parsed['args']
        else:
            args = message.text.split()[1:]  # Fallback para método original

        # Listas pré-definidas (mesmas do screening)
        predefined_lists = {
            'açõesbr': [
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
            'açõeseua': [
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
            'criptos': [
                "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD",
                "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "SHIB-USD",
                "TRX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
                "FIL-USD", "APT-USD", "ARB-USD", "NEAR-USD", "VET-USD"
            ],
            'forex': ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X"],
            'commodities': ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "KE=F", "CC=F", "KC=F"]
        }

        if not args:
            help_message = """
📊 *DETECÇÃO DE TOPOS E FUNDOS*

📝 *Como usar:*
/topos_fundos [lista/ativos]

📊 *Listas pré-definidas:*
• açõesBR - Ações brasileiras
• açõesEUA - Ações americanas
• criptos - Criptomoedas
• forex - Pares de moedas
• commodities - Commodities

⏰ *Configurações fixas:*
• Timeframe: 1 dia (fixo)
• Período: 2 anos de dados históricos

📈 *Exemplos:*
`/topos_fundos açõesBR`
`/topos_fundos açõesEUA`
`/topos_fundos criptos`
`/topos_fundos BTC-USD ETH-USD PETR4.SA VALE3.SA`

🎯 *O que detecta:*
• Possíveis fundos (oportunidades de compra)
• Possíveis topos (oportunidades de venda)
• Baseado em Bollinger Bands
            """
            bot.reply_to(message, help_message, parse_mode='Markdown')
            return

        symbols = []
        
        # Verificar se é uma lista pré-definida ou ativos individuais
        if len(args) == 1 and args[0].lower() in predefined_lists:
            list_name = args[0].lower()
            symbols = predefined_lists[list_name]
            list_display_name = {
                'açõesbr': 'Ações Brasileiras',
                'açõeseua': 'Ações Americanas', 
                'criptos': 'Criptomoedas',
                'forex': 'Forex',
                'commodities': 'Commodities'
            }
            bot.reply_to(message, f"📊 Analisando topos e fundos: {list_display_name[list_name]} ({len(symbols)} ativos)", parse_mode='Markdown')
        else:
            symbols = args

        if not symbols:
            bot.reply_to(message, "❌ Por favor, forneça uma lista válida ou pelo menos um ativo para análise.", parse_mode='Markdown')
            return

        # Limitação para evitar timeout
        if len(symbols) > 50:
            bot.reply_to(message, f"⚠️ Lista muito grande ({len(symbols)} ativos). Analisando os primeiros 200 ativos...", parse_mode='Markdown')
            symbols = symbols[:200]

        bot.reply_to(message, f"🔄 Analisando topos e fundos para {len(symbols)} ativos...", parse_mode='Markdown')

        # Detectar topos e fundos
        results = trading_bot.detect_tops_bottoms(symbols)

        if results:
            # Data atual da análise
            data_analise = datetime.now().strftime("%d/%m/%Y")
            
            response = f"📊 *DETECÇÃO DE TOPOS E FUNDOS*\n📅 {data_analise}\n\n⏰ Timeframe: 1 dia (fixo)\n📅 Período: 2 anos de dados\n📈 Total analisado: {len(symbols)} ativos\n\n"

            buy_opportunities = [r for r in results if 'Compra' in r['signal']]
            sell_opportunities = [r for r in results if 'Venda' in r['signal']]

            if buy_opportunities:
                response += "🟢 *POSSÍVEIS FUNDOS (COMPRA):*\n"
                for result in buy_opportunities:
                    response += f"• *{result['symbol']}*: {result['current_price']:.2f}\n"
                    response += f"  📊 Distância: {result['distance_pct']:.2f}%\n\n"

            if sell_opportunities:
                response += "🔴 *POSSÍVEIS TOPOS (VENDA):*\n"
                for result in sell_opportunities:
                    response += f"• *{result['symbol']}*: {result['current_price']:.2f}\n"
                    response += f"  📊 Distância: {result['distance_pct']:.2f}%\n\n"

            # Dividir mensagem se muito longa
            if len(response) > 4000:
                parts = response.split('🔴 *POSSÍVEIS TOPOS (VENDA):*')
                if len(parts) > 1:
                    # Enviar fundos primeiro
                    first_part = parts[0]
                    bot.reply_to(message, first_part, parse_mode='Markdown')
                    # Enviar topos depois
                    second_part = "🔴 *POSSÍVEIS TOPOS (VENDA):*" + parts[1]
                    bot.reply_to(message, second_part, parse_mode='Markdown')
                else:
                    bot.reply_to(message, response, parse_mode='Markdown')
            else:
                bot.reply_to(message, response, parse_mode='Markdown')

            logger.info(f"Topos e fundos enviados para {user_name}: {len(results)} oportunidades de {len(symbols)} ativos")
        else:
            bot.reply_to(message, f"ℹ️ Nenhuma oportunidade de topo ou fundo detectada nos {len(symbols)} ativos analisados.", parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Erro no comando /topos_fundos: {str(e)}")
        bot.reply_to(message, "❌ Erro ao processar topos e fundos. Tente novamente.")

@bot.message_handler(commands=['status'])
def status_command(message):
    try:
        logger.info(f"Comando /status recebido de {message.from_user.first_name}")

        status_message = """
📊 *STATUS DO BOT*

🤖 Bot: Online ✅
⏰ Timeframe: 1 dia
📅 Período análise: 365 dias
🔄 Última verificação: """ + datetime.now().strftime("%d/%m/%Y %H:%M") + """

🎯 *Estratégias disponíveis:*
• Agressiva 🔥
• Balanceada ⚖️
• Conservadora 🛡️

📈 *Funcionalidades ativas:*
• Screening de ativos ✅
• Detecção topos/fundos ✅
• Alertas em tempo real ✅
        """
        bot.reply_to(message, status_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Erro no comando /status: {str(e)}")
        bot.reply_to(message, "❌ Erro ao verificar status.")

@bot.message_handler(commands=['analise'])
def analise_command(message):
    try:
        user_name = message.from_user.first_name
        logger.info(f"Comando /analise recebido de {user_name}")

        # Parse arguments with fuzzy matching
        parsed = parse_flexible_command(message.text)
        if parsed and parsed['command'] == 'analise':
            args = parsed['args']
        else:
            args = message.text.split()[1:]  # Fallback para método original

        if len(args) < 3:
            help_message = """📊 ANÁLISE INDIVIDUAL DE ATIVO

📝 Como usar:
/analise [estrategia] [ativo] [timeframe] [modelo] [data_inicio] [data_fim]

🎯 Estratégias disponíveis:
• agressiva - Mais sinais, maior frequência
• balanceada - Equilibrada (recomendada)
• conservadora - Sinais mais confiáveis

🤖 Modelos disponíveis:
• ovelha - Modelo clássico (padrão)
• ovelha2 - Modelo com Machine Learning

⏰ Timeframes disponíveis:
1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk

📅 Formato de datas (opcional):
YYYY-MM-DD (exemplo: 2024-01-01)

📈 Exemplos:
/analise balanceada PETR4.SA 1d
/analise agressiva BTC-USD 4h ovelha2
/analise conservadora AAPL 1d ovelha 2024-06-01 2024-12-01

💡 Ativos suportados:
• Cripto: BTC-USD, ETH-USD, etc.
• Ações BR: PETR4.SA, VALE3.SA, etc.
• Ações US: AAPL, GOOGL, etc.
• Forex: EURUSD=X, etc.

ℹ️ Se não especificar modelo, será usado OVELHA clássico
ℹ️ Se não especificar datas, será usado período padrão baseado no timeframe"""
            bot.reply_to(message, help_message)
            return

        strategy_input = args[0].lower()
        symbol = args[1].upper()
        timeframe = args[2].lower()

        # Modelo opcional (4º argumento)
        model_input = "ovelha"  # padrão
        start_date = None
        end_date = None

        # Verificar se o 4º argumento é um modelo
        if len(args) >= 4:
            if args[3].lower() in ['ovelha', 'ovelha2']:
                model_input = args[3].lower()
                # Datas começam no 5º argumento
                if len(args) >= 6:
                    try:
                        start_date = args[4]
                        end_date = args[5]
                        datetime.strptime(start_date, '%Y-%m-%d')
                        datetime.strptime(end_date, '%Y-%m-%d')
                    except ValueError:
                        bot.reply_to(message, "❌ Formato de data inválido. Use YYYY-MM-DD (exemplo: 2024-01-01)")
                        return
            else:
                # 4º argumento não é modelo, deve ser data
                try:
                    start_date = args[3]
                    end_date = args[4] if len(args) >= 5 else None
                    if start_date:
                        datetime.strptime(start_date, '%Y-%m-%d')
                    if end_date:
                        datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    bot.reply_to(message, "❌ Formato de data inválido. Use YYYY-MM-DD (exemplo: 2024-01-01)")
                    return

        # Mapear estratégias
        strategy_map = {
            'agressiva': 'Agressivo',
            'balanceada': 'Balanceado', 
            'conservadora': 'Conservador'
        }

        if strategy_input not in strategy_map:
            bot.reply_to(message, "❌ Estratégia inválida. Use: agressiva, balanceada ou conservadora")
            return

        strategy = strategy_map[strategy_input]

        # Validar timeframes
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk']
        if timeframe not in valid_timeframes:
            bot.reply_to(message, f"❌ Timeframe inválido. Use: {', '.join(valid_timeframes)}")
            return

        model_display = "OVELHA V2" if model_input == "ovelha2" else "OVELHA"
        
        if start_date and end_date:
            bot.reply_to(message, f"🔄 Analisando {symbol} de {start_date} até {end_date} com modelo {model_display} e estratégia {strategy_input} no timeframe {timeframe}...")
        else:
            bot.reply_to(message, f"🔄 Analisando {symbol} com modelo {model_display} e estratégia {strategy_input} no timeframe {timeframe}...")

        # Gerar análise e gráfico
        chart_result = trading_bot.generate_analysis_chart(symbol, strategy, timeframe, model_input, start_date, end_date)

        if chart_result['success']:
            # Enviar gráfico
            with open(chart_result['chart_path'], 'rb') as chart_file:
                bot.send_photo(
                    message.chat.id, 
                    chart_file,
                    caption=chart_result['caption'],
                    parse_mode='HTML'
                )

            # Limpar arquivo temporário
            import os
            os.remove(chart_result['chart_path'])

            logger.info(f"Análise enviada para {user_name}: {symbol}")
        else:
            bot.reply_to(message, f"❌ {chart_result['error']}")

    except Exception as e:
        logger.error(f"Erro no comando /analise: {str(e)}")
        bot.reply_to(message, "❌ Erro ao processar análise. Verifique os parâmetros e tente novamente.")

@bot.message_handler(commands=['restart'])
def restart_command(message):
    try:
        user_name = message.from_user.first_name
        logger.info(f"Comando /restart recebido de {user_name}")

        restart_message = """🔄 REINICIANDO BOT...

⚠️ O bot será reiniciado completamente.
⏳ Aguarde alguns segundos e tente novamente.

🤖 Status: Reiniciando sistema...
📡 Reconectando aos serviços...
🔧 Limpando cache e memória...

✅ O bot voltará online em instantes!"""

        bot.reply_to(message, restart_message)
        logger.info(f"Mensagem de restart enviada para {user_name}")

        # Aguardar um pouco para enviar a mensagem antes de reiniciar
        time.sleep(2)

        # Parar o bot e reiniciar o processo
        logger.info("🔄 Reiniciando bot por comando do usuário...")
        bot.stop_polling()

        # Importar os módulos necessários para reiniciar
        import os
        import sys

        # Reiniciar o processo Python
        logger.info("🚀 Executando restart completo...")
        os.execv(sys.executable, ['python'] + sys.argv)

    except Exception as e:
        logger.error(f"Erro no comando /restart: {str(e)}")
        bot.reply_to(message, "❌ Erro ao reiniciar o bot. Tente novamente.")

@bot.message_handler(commands=['help'])
def help_command(message):
    try:
        logger.info(f"Comando /help recebido de {message.from_user.first_name}")

        help_message = """🤖 AJUDA - OVECCHIA TRADING BOT

📋 COMANDOS DISPONÍVEIS:

🏠 /start - Iniciar o bot

📊 /analise [estrategia] [ativo] [timeframe] [modelo] [data_inicio] [data_fim]
   Exemplo: /analise balanceada PETR4.SA 1d
   Com modelo: /analise balanceada PETR4.SA 1d ovelha2
   Com datas: /analise balanceada PETR4.SA 1d ovelha 2024-01-01 2024-06-01
   ⚠️ Timeframes personalizáveis: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk

🔍 /screening [estrategia] [lista/ativos]
   Com lista: /screening balanceada açõesBR
   Individual: /screening balanceada BTC-USD ETH-USD
   ⚠️ Timeframe fixo: 1d | Período fixo: 2 anos

📈 /topos_fundos [lista/ativos]
   Com lista: /topos_fundos açõesEUA
   Individual: /topos_fundos PETR4.SA VALE3.SA
   ⚠️ Timeframe fixo: 1d | Período fixo: 2 anos

📊 /status - Ver status do bot

🔄 /restart - Reiniciar o bot (em caso de problemas)

❓ /help - Esta mensagem de ajuda

🎯 ESTRATÉGIAS:
• agressiva - Mais sinais
• balanceada - Equilibrada
• conservadora - Mais confiável

🤖 MODELOS:
• ovelha - Modelo clássico (padrão)
• ovelha2 - Machine Learning (Random Forest)

📊 LISTAS PRÉ-DEFINIDAS:
• açõesBR - Ações brasileiras (126 ativos)
• açõesEUA - Ações americanas (100+ ativos)
• criptos - Criptomoedas principais (20 ativos)
• forex - Pares de moedas (8 pares)
• commodities - Commodities (10 ativos)

⏰ TIMEFRAMES:
• /analise: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk (personalizável)
• /screening: 1d (fixo) - 2 anos de dados
• /topos_fundos: 1d (fixo) - 2 anos de dados

💡 EXEMPLOS:
• /screening balanceada açõesBR
• /topos_fundos criptos
• /analise agressiva NVDA 4h
• /analise balanceada PETR4.SA 1d ovelha2"""
        bot.reply_to(message, help_message)
    except Exception as e:
        logger.error(f"Erro no comando /help: {str(e)}")
        bot.reply_to(message, "❌ Erro ao exibir ajuda.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_message = message.text
        user_name = message.from_user.first_name

        logger.info(f"Mensagem recebida de {user_name}: {user_message}")

        # Tentar identificar comando com fuzzy matching
        parsed = parse_flexible_command(user_message)
        
        if parsed:
            command = parsed['command']
            logger.info(f"Comando fuzzy identificado: {command} (original: {parsed['original_text']})")
            
            # Redirecionar para o handler apropriado
            if command == 'start':
                start_command(message)
            elif command == 'analise':
                analise_command(message)
            elif command == 'screening':
                screening_command(message)
            elif command == 'topos_fundos':
                topos_fundos_command(message)
            elif command == 'status':
                status_command(message)
            elif command == 'restart':
                restart_command(message)
            elif command == 'help':
                help_command(message)
            return
        
        # Mensagens de saudação
        user_message_lower = user_message.lower()
        if any(word in user_message_lower for word in ['oi', 'olá', 'hello', 'hi']):
            bot.reply_to(message, "👋 Olá! Use /help para ver os comandos disponíveis.")
        elif any(word in user_message_lower for word in ['ajuda', 'help']):
            help_command(message)
        else:
            bot.reply_to(message, "🤖 Use /help para ver os comandos disponíveis.\n\n💡 Dica: Você pode digitar comandos mesmo com pequenos erros de digitação!")

    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {str(e)}")

def run_bot():
    """Função para rodar o bot"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.info("🤖 Iniciando OVECCHIA TRADING BOT...")
            print("🤖 OVECCHIA TRADING BOT ONLINE!")

            # Configurar comandos do bot
            bot.set_my_commands([
                telebot.types.BotCommand("start", "Iniciar o bot"),
                telebot.types.BotCommand("analise", "Análise individual com gráfico"),
                telebot.types.BotCommand("screening", "Screening de múltiplos ativos"),
                telebot.types.BotCommand("topos_fundos", "Detectar topos e fundos"),
                telebot.types.BotCommand("status", "Ver status do bot"),
                telebot.types.BotCommand("restart", "Reiniciar o bot"),
                telebot.types.BotCommand("help", "Ajuda com comandos")
            ])

            logger.info("🤖 Bot iniciado com sucesso!")

            # Rodar o bot
            bot.polling(none_stop=True, interval=2, timeout=30)

        except Exception as e:
            retry_count += 1
            logger.error(f"Erro crítico no bot (tentativa {retry_count}/{max_retries}): {str(e)}")
            print(f"❌ Erro ao iniciar bot (tentativa {retry_count}/{max_retries}): {str(e)}")

            if retry_count < max_retries:
                wait_time = 5 * retry_count  # Aumentar tempo de espera a cada tentativa
                logger.info(f"🔄 Tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)
            else:
                logger.error("🛑 Máximo de tentativas excedido. Bot será encerrado.")
                break

if __name__ == '__main__':
    try:
        run_bot()
    except KeyboardInterrupt:
        logger.info("Bot interrompido pelo usuário")
        print("🛑 Bot interrompido")
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        print(f"💥 Erro fatal: {str(e)}")