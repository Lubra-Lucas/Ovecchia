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
import requests
import ccxt
import schedule
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot token - usar variável de ambiente para segurança
import os
BOT_TOKEN = os.environ.get('BOT_TOKEN', "8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k")

# Initialize bot with error handling
try:
    bot = telebot.TeleBot(BOT_TOKEN, threaded=False, skip_pending=True)
    logger.info("🤖 Bot do Telegram inicializado com sucesso")
except Exception as e:
    logger.error(f"❌ Erro ao inicializar bot do Telegram: {str(e)}")
    raise

# Thread lock para evitar processamento simultâneo
import threading
request_lock = threading.Lock()
user_locks = {}  # Lock por usuário

# Helper function to safely reply to messages
def safe_bot_reply(message, text, parse_mode=None):
    """Safely replies to a message, handling potential API errors."""
    try:
        bot.reply_to(message, text, parse_mode=parse_mode)
    except telebot.apihelper.ApiTelegramException as e:
        logger.error(f"Telegram API error: {e}")
        # Handle specific errors if necessary, e.g., message too long
        if "message is too long" in str(e):
            parts = text.split('\n')
            current_part = ""
            for part in parts:
                if len(current_part) + len(part) + 1 < 4096:
                    current_part += part + "\n"
                else:
                    try:
                        bot.reply_to(message, current_part, parse_mode=parse_mode)
                    except:
                        pass # Ignore if even sending parts fails
                    current_part = part + "\n"
            if current_part:
                try:
                    bot.reply_to(message, current_part, parse_mode=parse_mode)
                except:
                    pass
        else:
            # For other API errors, maybe send a generic message
            try:
                bot.reply_to(message, "❌ Ocorreu um erro ao processar sua solicitação. Tente novamente.")
            except:
                pass # Ignore if sending generic message fails too
    except Exception as e:
        logger.error(f"Unexpected error in safe_bot_reply: {str(e)}")
        # Generic fallback for non-API errors
        try:
            bot.reply_to(message, "❌ Ocorreu um erro inesperado. Tente novamente.")
        except:
            pass

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
        'analise': ['analise', 'analisar', 'analysis', 'analyze', 'grafico', 'chart'],
        'screening': ['screening', 'screnning', 'screning', 'screen', 'varredura', 'busca'],
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
        self.active_alerts = {}  # {user_id: {'symbols': [], 'source': '', 'model': '', 'strategy': '', 'timeframe': '', 'chat_id': ''}}
        self.alert_states = {}  # {user_id: {symbol: last_state}}
        self.active_tasks = {}  # {user_id: {'task_type': '', 'start_time': datetime, 'thread': None}}
        self.paused_users = set()  # Usuários que pausaram operações
        self.processing_users = set()  # Usuários sendo processados atualmente
        self.user_locks = {}  # Locks individuais por usuário
    
    def get_user_lock(self, user_id):
        """Obtém ou cria um lock para o usuário específico"""
        if user_id not in self.user_locks:
            self.user_locks[user_id] = threading.Lock()
        return self.user_locks[user_id]

    def get_ccxt_data(self, symbol, interval="1d", limit=1000):
        """Função para coletar dados usando CCXT com timeout otimizado"""
        try:
            # Configuração mais agressiva de timeout para timeframes pequenos
            timeout_ms = 15000 if interval in ['1m', '5m', '15m', '30m'] else 30000

            exchange = ccxt.binanceus({
                'enableRateLimit': True,
                'timeout': timeout_ms,
                'rateLimit': 2000,  # Rate limit mais agressivo
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })

            # Normalizar símbolo para formato CCXT
            ccxt_symbol = symbol.upper()

            # Conversões de formato
            if '-USD' in ccxt_symbol:
                ccxt_symbol = ccxt_symbol.replace('-USD', '/USDT')
            elif '-USDT' in ccxt_symbol:
                ccxt_symbol = ccxt_symbol.replace('-USDT', '/USDT')
            elif '/' not in ccxt_symbol:
                # Se não tem barra, assumir que precisa de /USDT
                ccxt_symbol = ccxt_symbol + '/USDT'

            # Verificar se o símbolo existe na exchange
            markets = exchange.load_markets()
            if ccxt_symbol not in markets:
                logger.error(f"Símbolo {ccxt_symbol} não encontrado na Binance")
                return pd.DataFrame()

            # Validar timeframe
            if interval not in exchange.timeframes:
                logger.error(f"Timeframe {interval} não suportado pela Binance")
                return pd.DataFrame()

            # Ajustar limite drasticamente baseado no timeframe para evitar timeout
            if interval in ['1m', '5m']:
                limit = min(200, limit)  # Reduzido para 200
            elif interval in ['15m', '30m']:
                limit = min(300, limit)  # Reduzido para 300
            elif interval in ['1h', '4h']:
                limit = min(500, limit)  # Máximo 500
            else:
                limit = min(1000, limit)  # Máximo 1000 para timeframes maiores

            logger.info(f"Coletando {limit} registros de {ccxt_symbol} no timeframe {interval} (timeout: {timeout_ms}ms)")

            # Implementar timeout manual usando threading
            import threading
            result = {'data': None, 'error': None}

            def fetch_data():
                try:
                    result['data'] = exchange.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=limit)
                except Exception as e:
                    result['error'] = str(e)

            # Iniciar thread com timeout
            thread = threading.Thread(target=fetch_data)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_ms/1000)  # Converter para segundos

            if thread.is_alive():
                logger.error(f"Timeout ao coletar dados CCXT para {ccxt_symbol} após {timeout_ms/1000}s")
                return pd.DataFrame()

            if result['error']:
                logger.error(f"Erro durante coleta CCXT: {result['error']}")
                return pd.DataFrame()

            ohlcv = result['data']
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"Nenhum dado OHLCV retornado para {ccxt_symbol}")
                return pd.DataFrame()

            # Criar DataFrame
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])

            # Converter timestamp para datetime
            df['time'] = pd.to_datetime(df['time'], unit='ms')

            # Garantir que os tipos numéricos estão corretos
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

            # Verificar se há dados válidos
            if df['close'].isna().all():
                logger.error(f"Todos os preços de fechamento são NaN para {ccxt_symbol}")
                return pd.DataFrame()

            # Ordenar por tempo
            df = df.sort_values("time").reset_index(drop=True)

            logger.info(f"Dados CCXT coletados com sucesso para {ccxt_symbol}: {len(df)} registros")
            return df

        except ccxt.NetworkError as e:
            logger.error(f"Erro de rede ao acessar CCXT para {symbol}: {str(e)}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            logger.error(f"Erro da exchange CCXT para {symbol}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro geral ao coletar dados CCXT para {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_twelve_data_data(self, symbol, start_date, end_date, interval="1d", limit=2000):
        """Função para coletar dados usando TwelveData API"""
        try:
            logger.info(f"Coletando dados para {symbol} via 12Data com intervalo {interval}")

            # Sua chave da Twelve Data
            API_KEY = "8745d2a910c841e4913afc40a6368dcb"

            # Usar o símbolo exatamente como o usuário digitou
            # TwelveData espera o formato "BTC/USD", "ETH/USD", etc.
            processed_symbol = symbol

            # Mapear timeframes do Telegram para 12Data
            twelve_interval_map = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '1h',
                '4h': '4h',
                '1d': '1day',
                '1wk': '1week'
            }
            twelve_interval = twelve_interval_map.get(interval.lower())
            if not twelve_interval:
                logger.error(f"Timeframe inválido para 12Data: {interval}")
                return pd.DataFrame()

            # Endpoint para pegar dados com quantidade configurável
            url = f"https://api.twelvedata.com/time_series?symbol={processed_symbol}&interval={twelve_interval}&apikey={API_KEY}&outputsize={min(limit, 5000)}"

            logger.info(f"Fazendo requisição para 12Data: {url}")

            # Faz a requisição
            response = requests.get(url, timeout=30).json()

            # Verifica se houve erro
            if "values" not in response:
                error_msg = response.get('message', 'Erro desconhecido')
                logger.error(f"Erro na API TwelveData: {error_msg}")
                return pd.DataFrame()

            # Cria o DataFrame
            df = pd.DataFrame(response['values'])

            if df.empty:
                logger.warning(f"Nenhum dado retornado pela TwelveData para {symbol}")
                return pd.DataFrame()

            # Converte colunas
            df['datetime'] = pd.to_datetime(df['datetime'])
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

            # Adicionar coluna volume se não existir
            if 'volume' not in df.columns:
                df['volume'] = 0.0
            else:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)

            # Ordena do mais antigo para o mais recente
            df = df.sort_values(by='datetime').reset_index(drop=True)

            # Padronizar nomes das colunas
            df.rename(columns={'datetime': 'time'}, inplace=True)

            # Verificar se há dados válidos
            if df['close'].isna().all():
                logger.error(f"Todos os preços de fechamento são NaN para {symbol}")
                return pd.DataFrame()

            logger.info(f"Dados 12Data coletados com sucesso para {symbol}: {len(df)} registros de {df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')} até {df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
            return df

        except Exception as e:
            logger.error(f"Erro ao buscar dados via TwelveData para {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_market_data(self, symbol, start_date, end_date, interval="1d", data_source="yahoo"):
        """Função para coletar dados do mercado"""
        try:
            logger.info(f"Coletando dados para {symbol} via {data_source}")

            # Mapear para fonte correta
            if data_source == "ccxt":
                df = self.get_ccxt_data(symbol, interval, 1000)
            elif data_source == "twelvedata":
                df = self.get_twelve_data_data(symbol, start_date, end_date, interval, 1000)
            else: # Yahoo Finance
                try:
                    # Yahoo Finance interval mapping
                    yf_interval_map = {
                        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                        '1h': '1h', '4h': '4h', '1d': '1d', '1wk': '1wk', '1mo': '1mo'
                    }
                    yf_interval = yf_interval_map.get(interval.lower())
                    if not yf_interval:
                        logger.info(f"Timeframe {interval} não suportado pelo Yahoo Finance. Usando '1d'.")
                        yf_interval = '1d'

                    # Se o intervalo for muito pequeno e não for 1m, 4h, etc, pode não ser suportado
                    if interval not in ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo']:
                        logger.warning(f"Timeframe {interval} não suportado pelo Yahoo Finance. Usando '1d'.")
                        yf_interval = '1d'

                    df = yf.download(symbol, start=start_date, end=end_date, interval=yf_interval, progress=False)

                    if df is None or df.empty:
                        logger.warning(f"Yahoo Finance: Sem dados para {symbol}")
                        return pd.DataFrame()

                    # Handle multi-level columns if present
                    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                        try:
                            df = df.xs(symbol, level='Ticker', axis=1, drop_level=True)
                        except KeyError:
                            # Se não conseguir extrair por ticker, usar o primeiro nível
                            df.columns = df.columns.droplevel(1)

                    df.reset_index(inplace=True)

                    # Standardize column names
                    column_mapping = {
                        "Datetime": "time",
                        "Date": "time",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "close", # Use Adj Close if available
                        "Volume": "volume"
                    }
                    df.rename(columns=column_mapping, inplace=True)

                    # Garantir que as colunas essenciais existam e estejam com tipos corretos
                    for col in ['time', 'open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            df[col] = 0.0
                        elif col == 'time':
                            # Converter para datetime se necessário
                            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                                df['time'] = pd.to_datetime(df['time'])
                        else:
                            df[col] = pd.to_numeric(df[col], errors='coerce') # Converte para numérico, erros viram NaN

                    # Remover linhas com valores NaN nas colunas essenciais após conversão
                    df = df.dropna(subset=['close', 'open', 'high', 'low', 'volume'])

                    logger.info(f"Dados Yahoo coletados com sucesso para {symbol}: {len(df)} registros")
                    return df

                except Exception as e:
                    logger.error(f"Erro específico do Yahoo Finance para {symbol}: {str(e)}")
                    return pd.DataFrame()

        except Exception as e:
            logger.error(f"Erro geral ao coletar dados para {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_ovelha_v2_signals(
        self,
        df,
        strategy_type="Balanceado",
        sma_short=60,
        sma_long=70,
        lookahead=3,
        # ----- THRESHOLD -----
        use_dynamic_threshold=True,
        vol_factor=0.5,          # multiplicador do ATR_rel (ATR/close) para o threshold adaptativo
        threshold_fixed=0.0003,  # fallback caso use_dynamic_threshold=False
        # ----- RF -----
        n_estimators=200,
        max_depth=None,
        class_weight='balanced',   # ajuda no desbalanceamento das classes
        random_state=42
    ):
        """
        Função para calcular sinais usando o modelo OVELHA V2 com Random Forest (Versão Aprimorada)

        Nova versão com:
        - Novas features: ATR_7, stddev_20, slope_SMA_long, MACD_hist
        - Threshold dinâmico baseado na volatilidade
        - Buffer adaptativo automático
        - Random Forest com balanceamento de classes
        """
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

            df_work = df.copy()

            # =======================
            # CÁLCULO DAS FEATURES
            # =======================
            df_work[f'SMA_{sma_short}'] = df_work['close'].rolling(window=sma_short).mean()
            df_work[f'SMA_{sma_long}']  = df_work['close'].rolling(window=sma_long).mean()
            df_work['SMA_20']           = df_work['close'].rolling(window=20).mean()

            # RSI(14)
            delta = df_work['close'].diff()
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
            avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df_work['RSI_14'] = 100 - (100 / (1 + rs))
            df_work['RSI_14'] = df_work['RSI_14'].bfill()

            # RSL(20)
            df_work['RSL_20'] = df_work['close'] / df_work['SMA_20']

            # ATR base (14)
            df_work['prior_close'] = df_work['close'].shift(1)
            df_work['tr1'] = df_work['high'] - df_work['low']
            df_work['tr2'] = (df_work['high'] - df_work['prior_close']).abs()
            df_work['tr3'] = (df_work['low'] - df_work['prior_close']).abs()
            df_work['TR']  = df_work[['tr1', 'tr2', 'tr3']].max(axis=1)
            df_work['ATR'] = df_work['TR'].rolling(window=14).mean()

            # 🔹NOVAS FEATURES
            # ATR_7 (volatilidade recente, mais sensível)
            df_work['ATR_7'] = df_work['TR'].rolling(window=7).mean()

            # Desvio padrão 20 dos retornos (ruído/aleatoriedade relativa)
            df_work['ret_1']     = df_work['close'].pct_change()
            df_work['stddev_20'] = df_work['ret_1'].rolling(window=20).std()

            # Slope da SMA longa (tendência/regime) - aprox. simples em janela 20
            _slope_w = 20
            sma_l = df_work[f'SMA_{sma_long}']
            df_work['slope_SMA_long'] = ((sma_l / sma_l.shift(_slope_w)) - 1) / _slope_w

            # MACD hist (12,26,9)
            ema12   = df_work['close'].ewm(span=12, adjust=False).mean()
            ema26   = df_work['close'].ewm(span=26, adjust=False).mean()
            macd    = ema12 - ema26
            signal  = macd.ewm(span=9, adjust=False).mean()
            df_work['MACD_hist'] = macd - signal

            # Derivadas e normalizações já existentes
            df_work['accel']    = df_work['ret_1'].diff()
            df_work['decel']    = -df_work['accel']
            df_work['atr_norm'] = df_work['ATR'] / df_work['close']

            # ===== BUFFER ADAPTATIVO =====
            b = 0.8  # multiplicador inicial (tune na otimização)
            df_work['buffer_pct'] = b * (df_work['ATR'] / df_work['close'])  # ou b * df_work['atr_norm']

            # (opcional) limitar extremos
            df_work['buffer_pct'] = df_work['buffer_pct'].clip(lower=0.0002, upper=0.005)  # 0.02% a 0.5%

            # =======================
            # LABEL (y) COM THRESHOLD
            # =======================
            df_work['future_ret'] = df_work['close'].shift(-lookahead) / df_work['close'] - 1

            if use_dynamic_threshold:
                # threshold adaptativo: vol_factor * (ATR / close)
                df_work['thr_used'] = vol_factor * (df_work['ATR'] / df_work['close'])
            else:
                df_work['thr_used'] = float(threshold_fixed)

            df_work['y'] = 0
            df_work.loc[df_work['future_ret'] >  df_work['thr_used'], 'y'] =  1
            df_work.loc[df_work['future_ret'] < -df_work['thr_used'], 'y'] = -1

            # Versão binária (apenas onde há trade)
            df_work['y_bin'] = df_work['y'].replace({0: np.nan})

            # =======================
            # TREINO RF (triclass)
            # =======================
            features = ['RSI_14', 'RSL_20', 'ATR', 'ATR_7', 'stddev_20', 'slope_SMA_long', 'MACD_hist', 'ret_1', 'accel', 'decel', 'atr_norm']
            mask_feat = df_work[features].notna().all(axis=1) & df_work['y'].notna()
            X = df_work.loc[mask_feat, features]
            y = df_work.loc[mask_feat, 'y']

            # Verificar se temos dados suficientes para treinar
            if len(X) < 50:
                logger.warning("Dados insuficientes para OVELHA V2, usando modelo clássico")
                return None

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1
            )
            rf.fit(X, y)

            # Previsão (triclass)
            df_work['Signal_model'] = np.nan
            df_work.loc[mask_feat, 'Signal_model'] = rf.predict(X)

            # Versão binária da previsão (apenas ±1; onde previu 0 vira NaN)
            df_work['Signal_model_bin'] = df_work['Signal_model'].replace({0: np.nan})

            # =======================
            # FILTRO DE TENDÊNCIA + HISTERESE (com buffer adaptativo)
            # =======================
            df_work['Signal'] = 'Stay Out'
            for i in range(1, len(df_work)):
                prev_estado = df_work['Signal'].iloc[i-1]
                price = df_work['close'].iloc[i]
                sma_s = df_work[f'SMA_{sma_short}'].iloc[i]
                sma_l = df_work[f'SMA_{sma_long}'].iloc[i]
                sm    = df_work['Signal_model'].iloc[i]
                buf   = df_work['buffer_pct'].iloc[i]  # <-- buffer dinâmico

                if sm == 1:
                    if price > sma_s * (1 + buf) and price > sma_l * (1 + buf):
                        df_work.iat[i, df_work.columns.get_loc('Signal')] = 'Buy'
                    else:
                        df_work.iat[i, df_work.columns.get_loc('Signal')] = prev_estado
                elif sm == -1:
                    if price < sma_s * (1 - buf):
                        df_work.iat[i, df_work.columns.get_loc('Signal')] = 'Sell'
                    else:
                        df_work.iat[i, df_work.columns.get_loc('Signal')] = prev_estado
                else:
                    df_work.iat[i, df_work.columns.get_loc('Signal')] = prev_estado

            # Persistência de estado
            df_work['Estado'] = 'Stay Out'
            for i in range(1, len(df_work)):
                sig = df_work['Signal'].iloc[i]
                df_work.iat[i, df_work.columns.get_loc('Estado')] = sig if sig != 'Stay Out' else df_work['Estado'].iloc[i-1]

            return df_work

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

    def perform_automated_screening(self, user_id, symbols_list, source, model_type, strategy_type, timeframe):
        """Realiza screening automático e detecta mudanças de estado"""
        try:
            current_states = {}
            changes_detected = []
            successful_analyses = 0

            for symbol in symbols_list:
                try:
                    logger.info(f"Analisando {symbol} para usuário {user_id}")

                    # Validar símbolo antes de processar
                    if not symbol or len(symbol.strip()) == 0:
                        logger.warning(f"Símbolo vazio ou inválido: '{symbol}'")
                        continue

                    symbol = symbol.strip().upper()

                    if source == "ccxt":
                        df = self.get_ccxt_data(symbol, timeframe, 1000)
                    elif source == "12data":
                        end_date = datetime.now().date()
                        start_date = end_date - timedelta(days=365)
                        df = self.get_twelve_data_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), timeframe, 2000)
                    else: # Yahoo
                        end_date = datetime.now().date()
                        start_date = end_date - timedelta(days=365)
                        df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"),
                                                end_date.strftime("%Y-%m-%d"), timeframe, "yahoo")

                    if df.empty:
                        logger.warning(f"Sem dados disponíveis para {symbol} via {source}")
                        continue

                    # Verificar se há dados suficientes
                    if len(df) < 50:
                        logger.warning(f"Dados insuficientes para {symbol}: apenas {len(df)} registros")
                        continue

                    # Escolher modelo baseado na seleção do usuário
                    if model_type == "ovelha2":
                        df_with_signals = self.calculate_ovelha_v2_signals(df, strategy_type)
                        if df_with_signals is not None and not df_with_signals.empty:
                            df = df_with_signals
                        else:
                            logger.info(f"Fallback para modelo clássico para {symbol}")
                            model_type = "ovelha"  # Fallback

                    if model_type == "ovelha" or 'Estado' not in df.columns:
                        df = self.calculate_indicators_and_signals(df, strategy_type)

                    if df.empty or 'Estado' not in df.columns:
                        logger.warning(f"Falha ao calcular indicadores para {symbol}")
                        continue

                    current_state = df['Estado'].iloc[-1]
                    current_price = df['close'].iloc[-1]

                    # Validar estado
                    if current_state not in ['Buy', 'Sell', 'Stay Out']:
                        logger.warning(f"Estado inválido para {symbol}: {current_state}")
                        continue

                    current_states[symbol] = {
                        'state': current_state,
                        'price': current_price
                    }
                    successful_analyses += 1

                    # Verificar se houve mudança de estado
                    if user_id in self.alert_states and symbol in self.alert_states[user_id]:
                        previous_state = self.alert_states[user_id][symbol]['state']
                        if current_state != previous_state:
                            changes_detected.append({
                                'symbol': symbol,
                                'previous_state': previous_state,
                                'current_state': current_state,
                                'current_price': current_price
                            })

                except Exception as e:
                    logger.error(f"Erro específico ao analisar {symbol}: {str(e)}")
                    continue

            # Atualizar estados salvos
            if user_id not in self.alert_states:
                self.alert_states[user_id] = {}
            self.alert_states[user_id].update(current_states)

            logger.info(f"Screening completado para usuário {user_id}: {successful_analyses}/{len(symbols_list)} símbolos analisados com sucesso")
            return current_states, changes_detected

        except Exception as e:
            logger.error(f"Erro geral no screening automatizado: {str(e)}")
            return {}, []

    def generate_analysis_chart(self, symbol, strategy_type, timeframe, model_type="ovelha", custom_start_date=None, custom_end_date=None, data_source="yahoo"):
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

            # Coletar dados baseado na fonte especificada
            if data_source == "ccxt":
                df = self.get_ccxt_data(symbol, timeframe, 1000)
            elif data_source == "twelvedata":
                df = self.get_twelve_data_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), timeframe, 2000)
            else: # Yahoo
                yf_interval_map = {
                    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                    '1h': '1h', '4h': '4h', '1d': '1d', '1wk': '1wk', '1mo': '1mo'
                }
                yf_interval = yf_interval_map.get(timeframe.lower())
                if not yf_interval:
                    yf_interval = '1d' # Default para timeframe desconhecido

                df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"),
                                        end_date.strftime("%Y-%m-%d"), timeframe, "yahoo")

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
            # Sanitizar nome do arquivo removendo caracteres especiais
            safe_symbol = symbol.replace('/', '_').replace('.', '_').replace('-', '_').replace('\\', '_').replace(':', '_')
            chart_filename = f"chart_{safe_symbol}_{int(datetime.now().timestamp())}.png"
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

@bot.message_handler(commands=['screening'])
def screening_command(message):
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "Usuário"
    
    # Obter lock do usuário
    user_lock = trading_bot.get_user_lock(user_id)
    
    if not user_lock.acquire(blocking=False):
        safe_bot_reply(message, "⏳ Você já tem uma operação em andamento. Aguarde terminar.")
        return
    
    try:
        logger.info(f"Comando /screening recebido de {user_name} (ID: {user_id})")
        
        # Verificar se usuário já está processando
        if user_id in trading_bot.processing_users:
            safe_bot_reply(message, "⏳ Processando comando anterior. Aguarde.")
            return
            
        # Marcar usuário como processando
        trading_bot.processing_users.add(user_id)

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
                            `/screening balanceada BTC-USD ETH-USD PETR4.SA VALE3.SA`

                            💡 *Nota:* Você pode usar listas pré-definidas OU especificar ativos individuais
                                        """
            safe_bot_reply(message, help_message, 'Markdown')
            return

        safe_bot_reply(message, "🔄 Processando screening...", 'Markdown')

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
            safe_bot_reply(message, f"📊 Analisando lista: {list_display_name[list_name]} ({len(symbols)} ativos)", 'Markdown')
        else:
            symbols = remaining_args

        if not symbols:
            safe_bot_reply(message, "❌ Por favor, forneça uma lista válida ou pelo menos um ativo para análise.", 'Markdown')
            return

        logger.info(f"Realizando screening para {len(symbols)} ativos com estratégia {strategy}")

        # Realizar screening (limitado a 50 ativos por vez para evitar timeout)
        if len(symbols) > 50:
            safe_bot_reply(message, f"⚠️ Lista muito grande ({len(symbols)} ativos). Analisando os ativos...", 'Markdown')
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
                    if len(current_message) + len(part) + 1 < 4096:
                        current_message += part + '\n\n'
                    else:
                        safe_bot_reply(message, current_message, 'Markdown')
                        current_message = part + '\n\n'

                if current_message.strip():
                    safe_bot_reply(message, current_message, 'Markdown')
            else:
                safe_bot_reply(message, response, 'Markdown')

            logger.info(f"Screening enviado para {user_name}: {len(results)} alertas de {len(symbols)} ativos")
        else:
            safe_bot_reply(message, f"ℹ️ Nenhuma mudança de estado detectada nos {len(symbols)} ativos analisados.", 'Markdown')
            logger.info(f"Nenhum alerta encontrado para {user_name}")

    except telebot.apihelper.ApiException as e:
        logger.error(f"Erro da API Telegram no /screening: {str(e)}")
        safe_bot_reply(message, "❌ Erro temporário da API. Aguarde alguns segundos e tente novamente.")
    except Exception as e:
        logger.error(f"Erro no comando /screening: {str(e)}")
        safe_bot_reply(message, "❌ Erro ao processar screening. Tente novamente.")
    finally:
        # Sempre limpar estados do usuário
        trading_bot.processing_users.discard(user_id)
        user_lock.release()





@bot.message_handler(commands=['analise'])
def analise_command(message):
    user_id = message.from_user.id
    user_name = message.from_user.first_name or "Usuário"
    
    # Obter lock do usuário para evitar processamento simultâneo
    user_lock = trading_bot.get_user_lock(user_id)
    
    if not user_lock.acquire(blocking=False):
        safe_bot_reply(message, "⏳ Você já tem uma operação em andamento. Aguarde terminar ou use /restart para limpar.")
        return
    
    try:
        logger.info(f"Comando /analise recebido de {user_name} (ID: {user_id})")
        
        # Verificar se usuário já está processando
        if user_id in trading_bot.processing_users:
            safe_bot_reply(message, "⏳ Processando comando anterior. Aguarde ou use /restart.")
            return
            
        # Marcar usuário como processando
        trading_bot.processing_users.add(user_id)

        # Verificar se usuário pausou operações
        if user_id in trading_bot.paused_users:
            trading_bot.paused_users.discard(user_id)

        # Verificar se já há uma tarefa ativa
        if user_id in trading_bot.active_tasks:
            active_task = trading_bot.active_tasks[user_id]
            duration = datetime.now() - active_task.get('start_time', datetime.now())

            if duration.seconds < 30:  # Menos de 30 segundos
                safe_bot_reply(message, "⏳ Já há uma análise em andamento. Aguarde ou use /pause para cancelar.")
                return
            elif duration.seconds < 120:  # Entre 30s e 2min
                safe_bot_reply(message, f"⚠️ Análise ativa há {duration.seconds}s. Use /pause para cancelar ou aguarde.")
                return
            else:
                # Tarefa travada há mais de 2 minutos, limpar e alertar
                del trading_bot.active_tasks[user_id]
                safe_bot_reply(message, f"⚠️ Tarefa anterior travada foi limpa. Iniciando nova análise...\n💡 Dica: Use timeframes maiores para evitar travamentos.")

        # Parse arguments with fuzzy matching
        parsed = parse_flexible_command(message.text)
        if parsed and parsed['command'] == 'analise':
            args = parsed['args']
        else:
            args = message.text.split()[1:]  # Fallback para método original

        # Argumentos esperados: [fonte] [estrategia] [ativo] [timeframe] [modelo] [data_inicio] [data_fim]
        if len(args) < 4: # Fonte, estratégia, ativo, timeframe são obrigatórios
            help_message = """
                            📊 ANÁLISE INDIVIDUAL DE ATIVO

                            📝 Como usar:
                            /analise [fonte] [estrategia] [ativo] [timeframe] [modelo] [data_inicio] [data_fim]

                            🔗 Fontes disponíveis:
                            • yahoo - Yahoo Finance (padrão)
                            • ccxt - Binance via CCXT (criptomoedas)
                            • twelvedata - 12Data (criptos, forex, ações)

                            🎯 Estratégias disponíveis:
                            • agressiva - Mais sinais, maior frequência
                            • balanceada - Equilibrada (recomendada)
                            • conservadora - Sinais mais confiáveis

                            🤖 Modelos disponíveis:
                            • ovelha - Modelo clássico (padrão)
                            • ovelha2 - Machine Learning (Random Forest)

                            ⏰ Timeframes disponíveis:
                            1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk

                            📅 Formato de datas (opcional):
                            YYYY-MM-DD (exemplo: 2024-01-01)

                            📈 Exemplos:
                            /analise yahoo balanceada PETR4.SA 1d
                            /analise twelvedata agressiva BTCUSDT 4h ovelha2
                            /analise yahoo conservadora AAPL 1d ovelha 2024-06-01 2024-12-01

                            💡 Ativos suportados:
                            • Yahoo: PETR4.SA, VALE3.SA, AAPL, BTC-USD, EURUSD=X
                            • CCXT: BTC/USDT, ETH/USDT, BNB/USDT
                            • 12Data: BTCUSDT, EURUSD, AAPL

                            ℹ️ Se não especificar fonte, será usado YAHOO
                            ℹ️ Se não especificar modelo, será usado OVELHA clássico
                            ℹ️ Se não especificar datas, será usado período padrão baseado no timeframe"""
            safe_bot_reply(message, help_message)
            return

        source_input = args[0].lower()
        strategy_input = args[1].lower()
        symbol = args[2].upper()
        timeframe = args[3].lower()

        # Modelo e datas são opcionais
        model_input = "ovelha"  # padrão
        start_date = None
        end_date = None

        # Verificar se o 4º argumento (após timeframe) é um modelo
        if len(args) >= 5:
            if args[4].lower() in ['ovelha', 'ovelha2']:
                model_input = args[4].lower()
                # Datas começam no 6º argumento
                if len(args) >= 7:
                    try:
                        start_date = args[5]
                        end_date = args[6]
                        datetime.strptime(start_date, '%Y-%m-%d')
                        datetime.strptime(end_date, '%Y-%m-%d')
                    except ValueError:
                        safe_bot_reply(message, "❌ Formato de data inválido. Use YYYY-MM-DD (exemplo: 2024-01-01)")
                        return
            else:
                # 5º argumento não é modelo, deve ser data
                try:
                    start_date = args[4]
                    end_date = args[5] if len(args) >= 6 else None
                    if start_date:
                        datetime.strptime(start_date, '%Y-%m-%d')
                    if end_date:
                        datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    safe_bot_reply(message, "❌ Formato de data inválido. Use YYYY-MM-DD (exemplo: 2024-01-01)")
                    return

        # Validar fonte
        if source_input not in ['yahoo', 'ccxt', 'twelvedata']:
            safe_bot_reply(message, "❌ Fonte inválida. Use: yahoo, ccxt ou twelvedata")
            return

        # Mapear estratégias
        strategy_map = {
            'agressiva': 'Agressivo',
            'balanceada': 'Balanceado',
            'conservadora': 'Conservador'
        }

        if strategy_input not in strategy_map:
            safe_bot_reply(message, "❌ Estratégia inválida. Use: agressiva, balanceada ou conservadora")
            return

        strategy = strategy_map[strategy_input]

        # Validar timeframes
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk']
        if timeframe not in valid_timeframes:
            safe_bot_reply(message, f"❌ Timeframe inválido. Use: {', '.join(valid_timeframes)}")
            return

        model_display = "OVELHA V2" if model_input == "ovelha2" else "OVELHA"

        # Registrar tarefa ativa
        trading_bot.active_tasks[user_id] = {
            'task_type': f'Análise {symbol} ({model_display})',
            'start_time': datetime.now(),
            'thread': None
        }

        # Aviso sobre tempo de processamento para timeframes menores
        warning_msg = ""
        if timeframe in ['1m', '5m', '15m', '30m'] and source_input == "ccxt":
            warning_msg = "\n⚠️ ATENÇÃO: Timeframes pequenos podem travar o bot! Recomendo usar 4h ou superior."

        if start_date and end_date:
            safe_bot_reply(message, f"🔄 Analisando {symbol} ({source_input}) de {start_date} até {end_date} com modelo {model_display} e estratégia {strategy_input} no timeframe {timeframe}...{warning_msg}")
        else:
            safe_bot_reply(message, f"🔄 Analisando {symbol} ({source_input}) com modelo {model_display} e estratégia {strategy_input} no timeframe {timeframe}...{warning_msg}")

        # Verificar se foi pausado antes de continuar
        if user_id in trading_bot.paused_users:
            if user_id in trading_bot.active_tasks:
                del trading_bot.active_tasks[user_id]
            trading_bot.processing_users.discard(user_id)
            safe_bot_reply(message, "⏸️ Análise cancelada pelo usuário.")
            return

        # Implementar timeout para análises que podem travar
        analysis_timeout = 30 if timeframe in ['1m', '5m', '15m', '30m'] and source_input == "ccxt" else 60

        def run_analysis():
            return trading_bot.generate_analysis_chart(symbol, strategy, timeframe, model_input, start_date, end_date, source_input)

        # Executar análise com timeout
        import threading
        result = {'chart_result': None, 'error': None, 'completed': False}

        def analysis_worker():
            try:
                result['chart_result'] = run_analysis()
                result['completed'] = True
            except Exception as e:
                result['error'] = str(e)
                result['completed'] = True

        # Iniciar thread da análise
        analysis_thread = threading.Thread(target=analysis_worker)
        analysis_thread.daemon = True
        analysis_thread.start()
        analysis_thread.join(timeout=analysis_timeout)

        # Verificar se completou
        if not result['completed']:
            # Timeout - limpar tarefa e informar usuário
            if user_id in trading_bot.active_tasks:
                del trading_bot.active_tasks[user_id]
            trading_bot.paused_users.add(user_id)
            trading_bot.processing_users.discard(user_id)

            safe_bot_reply(message, f"""⏰ **TIMEOUT - ANÁLISE CANCELADA**

🚨 A análise de {symbol} no timeframe {timeframe} demorou mais que {analysis_timeout}s e foi cancelada.

⚠️ **Problema comum:** Timeframes pequenos com CCXT frequentemente travam
🔧 **Solução:** Use /restart para limpar o bot completamente

🚀 **Alternativas que funcionam:**
• /analise yahoo balanceada BTC-USD 1d ovelha2 (via Yahoo)
• /analise twelvedata agressiva BTC/USD 4h ovelha
• Timeframes ≥ 4h são mais estáveis""", 'Markdown')

            logger.warning(f"Timeout na análise para {user_name}: {symbol} {timeframe}")
            return

        # Se chegou aqui, a análise completou
        if result['error']:
            chart_result = {'success': False, 'error': result['error']}
        else:
            chart_result = result['chart_result']

        # Remover tarefa ativa
        if user_id in trading_bot.active_tasks:
            del trading_bot.active_tasks[user_id]

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
            safe_bot_reply(message, f"❌ {chart_result['error']}")

    except telebot.apihelper.ApiException as e:
        # Limpar tarefa ativa em caso de erro
        if user_id in trading_bot.active_tasks:
            del trading_bot.active_tasks[user_id]
        logger.error(f"Erro da API Telegram no /analise: {str(e)}")
        safe_bot_reply(message, "❌ Erro temporário da API. Aguarde e tente novamente.")
    except Exception as e:
        # Limpar tarefa ativa em caso de erro
        if user_id in trading_bot.active_tasks:
            del trading_bot.active_tasks[user_id]
        logger.error(f"Erro no comando /analise: {str(e)}")
        safe_bot_reply(message, "❌ Erro ao processar análise. Tente novamente em alguns segundos.")
    finally:
        # Sempre limpar estados do usuário
        trading_bot.processing_users.discard(user_id)
        user_lock.release()

@bot.message_handler(commands=['screening_auto'])
def screening_auto_command(message):
    try:
        user_name = message.from_user.first_name
        user_id = message.from_user.id
        logger.info(f"Comando /screening_auto recebido de {user_name}")

        # Parse arguments
        args = message.text.split()[1:]

        if len(args) < 5: # fonte, símbolos, modelo, estratégia, timeframe são obrigatórios
            help_message = """
                            🔄 *SCREENING AUTOMÁTICO*

                            📝 *Como usar:*
                            /screening_auto [fonte] [símbolos] [modelo] [estrategia] [timeframe]

                            🔗 *Fontes disponíveis:*
                            • 12data - 12Data API (recomendado)
                            • yahoo - Yahoo Finance
                            • ccxt - Binance via CCXT

                            📊 *Símbolos:* Lista separada por vírgulas entre colchetes
                            • Para 12Data: [BTC/USD,ETH/USD,LTC/USD]
                            • Para Yahoo: [BTC-USD,ETH-USD,PETR4.SA]
                            • Para CCXT: [BTC/USDT,ETH/USDT,LTC/USDT]

                            🤖 *Modelos:*
                            • ovelha - Modelo clássico
                            • ovelha2 - Machine Learning (Random Forest)

                            🎯 *Estratégias:*
                            • agressiva - Mais sinais
                            • balanceada - Equilibrada
                            • conservadora - Mais confiáveis

                            ⏰ *Timeframes disponíveis:*
                            • 5m - 5 minutos (apenas 12Data)
                            • 15m - 15 minutos
                            • 1h - 1 hora
                            • 4h - 4 horas
                            • 1d - 1 dia (diário)

                            📈 *Exemplos:*
                            `/screening_auto 12data [BTC/USD,ETH/USD,LTC/USD] ovelha2 balanceada 4h`
                            `/screening_auto yahoo [BTC-USD,ETH-USD,PETR4.SA] ovelha balanceada 1d`
                            `/screening_auto ccxt [BTC/USDT,ETH/USDT,LTC/USDT] ovelha2 agressiva 4h`

                            💡 *Nota:* O bot enviará alertas no intervalo escolhido
                                        """
            safe_bot_reply(message, help_message, 'Markdown')
            return

        try:
            source = args[0].lower()
            symbols_str = args[1]
            model_type = args[2].lower()
            strategy = args[3].lower()
            timeframe = args[4].lower()

            # Validar fonte
            if source not in ['12data', 'twelvedata', 'yahoo', 'ccxt']:
                safe_bot_reply(message, "❌ Fonte inválida. Use: twelvedata , yahoo ou ccxt,")
                return

            # Normalizar fonte
            if source in ['12data', 'twelvedata']:
                source = '12data'

            # Extrair símbolos da lista
            if not symbols_str.startswith('[') or not symbols_str.endswith(']'):
                safe_bot_reply(message, "❌ Formato de símbolos inválido. Use: [SYMBOL1,SYMBOL2,...]")
                return

            symbols_list = [s.strip() for s in symbols_str[1:-1].split(',')]

            if len(symbols_list) == 0 or len(symbols_list) > 10:
                safe_bot_reply(message, "❌ Lista deve conter entre 1 e 10 símbolos")
                return

            # Validar modelo
            if model_type not in ['ovelha', 'ovelha2']:
                safe_bot_reply(message, "❌ Modelo inválido. Use: ovelha ou ovelha2")
                return

            # Validar estratégia
            strategy_map = {
                'agressiva': 'Agressivo',
                'balanceada': 'Balanceado',
                'conservadora': 'Conservador'
            }

            if strategy not in strategy_map:
                safe_bot_reply(message, "❌ Estratégia inválida. Use: agressiva, balanceada ou conservadora")
                return

            strategy_formatted = strategy_map[strategy]

            # Validar timeframe baseado na fonte
            if source == '12data':
                valid_timeframes = ['1m','5m', '15m', '1h', '4h', '1d']
            else:
                valid_timeframes = ['5m','15m', '1h', '4h', '1d']

            if timeframe not in valid_timeframes:
                safe_bot_reply(message, f"❌ Timeframe inválido para {source}. Use: {', '.join(valid_timeframes)}")
                return

            # Configurar alerta automático
            trading_bot.active_alerts[user_id] = {
                'symbols': symbols_list,
                'source': source,
                'model': model_type,
                'strategy': strategy_formatted,
                'timeframe': timeframe,
                'chat_id': message.chat.id
            }

            # Fazer primeira verificação
            safe_bot_reply(message, f"🔄 Configurando alerta automático...\n📊 {len(symbols_list)} símbolos via {source.upper()}\n⏰ Intervalo: {timeframe}")

            current_states, changes = trading_bot.perform_automated_screening(
                user_id, symbols_list, source, model_type, strategy_formatted, timeframe
            )

            # Verificar se conseguiu analisar pelo menos um símbolo
            if not current_states:
                format_examples = {
                    '12data': 'BTC/USD, ETH/USD, AAPL',
                    'yahoo': 'BTC-USD, ETH-USD, PETR4.SA, AAPL',
                    'ccxt': 'BTC/USDT, ETH/USDT, LTC/USDT'
                }

                error_message = f"""❌ **ERRO AO CONFIGURAR ALERTA**

                🔍 **Problema:** Nenhum dos símbolos pôde ser analisado via {source.upper()}.

                🔧 **Possíveis causas:**
                • Símbolos inválidos para a fonte {source.upper()}
                • Problemas de conectividade com a API
                • Timeframe {timeframe} não suportado para alguns símbolos

                💡 **Formato correto para {source.upper()}:**
                {format_examples.get(source, 'Verifique a documentação')}

                📝 **Exemplo correto:**
                `/screening_auto {source} [{format_examples.get(source, 'SYMBOL1,SYMBOL2').replace(', ', ',')}] {model_type} {strategy} {timeframe}`

                🔄 **Tente novamente** com símbolos válidos para a fonte escolhida."""
                safe_bot_reply(message, error_message, 'Markdown')
                return

            # Programar alertas baseado no timeframe
            schedule_alerts_for_user(user_id, timeframe)

            # Contar símbolos com sucesso e erro
            success_count = len(current_states)
            error_count = len(symbols_list) - success_count

            # Enviar confirmação
            confirmation_message = f"""✅ *ALERTA AUTOMÁTICO CONFIGURADO*

                📊 **Configuração:**
                🔗 Fonte: {source.upper()}
                🎯 Estratégia: {strategy}
                🤖 Modelo: {model_type.upper()}
                ⏰ Intervalo: {timeframe}

                📈 **Resultado:** {success_count}/{len(symbols_list)} símbolos válidos

                📊 **Símbolos monitorados:**
                """
            for symbol in symbols_list:
                if symbol in current_states:
                    state = current_states[symbol]['state']
                    price = current_states[symbol]['price']
                    state_icon = "🔵" if state == "Buy" else "🔴" if state == "Sell" else "⚫"
                    confirmation_message += f"• {symbol}: {state_icon} {state} ({price:.4f})\n"
                else:
                    confirmation_message += f"• {symbol}: ❌ Erro nos dados\n"

            if error_count > 0:
                confirmation_message += f"\n⚠️ **{error_count} símbolos com erro** - verifique os nomes"

            confirmation_message += f"\n🔔 Próximo alerta em: {timeframe}"
            confirmation_message += f"\n\n💡 **Dica:** Os símbolos são convertidos automaticamente para o formato da API (BTC/USD → btc-usd)"

            safe_bot_reply(message, confirmation_message, 'Markdown')
            logger.info(f"Alerta automático configurado para {user_name}: {len(symbols_list)} símbolos via {source}, {timeframe}")

        except Exception as e:
            logger.error(f"Erro ao processar argumentos: {str(e)}")
            safe_bot_reply(message, "❌ Erro ao processar comando. Verifique a sintaxe.")

    except Exception as e:
        logger.error(f"Erro no comando /screening_auto: {str(e)}")
        safe_bot_reply(message, "❌ Erro interno. Tente novamente.")

@bot.message_handler(commands=['stop_alerts'])
def stop_alerts_command(message):
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name
        logger.info(f"Comando /stop_alerts recebido de {user_name}")

        if user_id in trading_bot.active_alerts:
            del trading_bot.active_alerts[user_id]
            if user_id in trading_bot.alert_states:
                del trading_bot.alert_states[user_id]
            safe_bot_reply(message, "🛑 Alertas automáticos interrompidos com sucesso!")
            logger.info(f"Alertas interrompidos para {user_name}")
        else:
            safe_bot_reply(message, "ℹ️ Nenhum alerta automático ativo encontrado.")

    except Exception as e:
        logger.error(f"Erro no comando /stop_alerts: {str(e)}")
        safe_bot_reply(message, "❌ Erro ao interromper alertas.")

@bot.message_handler(commands=['list_alerts'])
def list_alerts_command(message):
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name
        logger.info(f"Comando /list_alerts recebido de {user_name} (ID: {user_id})")

        if user_id in trading_bot.active_alerts:
            alert_config = trading_bot.active_alerts[user_id]

            # Verificar se todas as chaves necessárias existem
            required_keys = ['symbols', 'source', 'strategy', 'model', 'timeframe']
            missing_keys = [key for key in required_keys if key not in alert_config]

            if missing_keys:
                logger.error(f"Chaves faltando na configuração de alerta para usuário {user_id}: {missing_keys}")
                safe_bot_reply(message, f"❌ Erro na configuração do alerta. Chaves faltando: {', '.join(missing_keys)}. Use /stop_alerts e configure novamente.")
                return

            # Validar se symbols é uma lista
            if not isinstance(alert_config['symbols'], list):
                logger.error(f"Campo 'symbols' não é uma lista para usuário {user_id}: {type(alert_config['symbols'])}")
                safe_bot_reply(message, "❌ Erro na configuração dos símbolos. Use /stop_alerts e configure novamente.")
                return

            symbols_list = ', '.join(alert_config['symbols'])

            # Construir mensagem de forma segura
            try:
                source = str(alert_config['source']).upper()
                strategy = str(alert_config['strategy'])
                model = str(alert_config['model']).upper()
                timeframe = str(alert_config['timeframe'])

                alert_info = f"""
                            📋 *ALERTA ATIVO*

                            🔗 Fonte: {source}
                            🎯 Estratégia: {strategy}
                            🤖 Modelo: {model}
                            ⏰ Intervalo: {timeframe}

                            📈 Símbolos ({len(alert_config['symbols'])}): {symbols_list}

                            🔔 Use /stop_alerts para interromper
                            """

                safe_bot_reply(message, alert_info, 'Markdown')
                logger.info(f"Lista de alertas enviada para {user_name}: {len(alert_config['symbols'])} símbolos")

            except Exception as format_error:
                logger.error(f"Erro ao formatar mensagem de alerta para usuário {user_id}: {str(format_error)}")
                # Enviar mensagem básica sem formatação
                basic_info = f"📋 ALERTA ATIVO\n\nFonte: {alert_config.get('source', 'N/A')}\nSímbolos: {len(alert_config.get('symbols', []))}\n\nUse /stop_alerts para interromper"
                safe_bot_reply(message, basic_info)

        else:
            safe_bot_reply(message, "ℹ️ Nenhum alerta automático ativo.")
            logger.info(f"Nenhum alerta ativo para {user_name}")

    except Exception as e:
        logger.error(f"Erro geral no comando /list_alerts para usuário {user_id}: {str(e)}")
        safe_bot_reply(message, "❌ Erro ao listar alertas. Tente novamente ou use /stop_alerts se houver problemas.")

@bot.message_handler(commands=['pause'])
def pause_command(message):
    """Comando para pausar operações em andamento"""
    try:
        user_name = message.from_user.first_name or "Usuário"
        user_id = message.from_user.id
        logger.info(f"Comando /pause recebido de {user_name} (ID: {user_id})")

        # Pausar usuário
        trading_bot.paused_users.add(user_id)
        trading_bot.processing_users.discard(user_id)
        
        # Limpar tarefas ativas
        if user_id in trading_bot.active_tasks:
            del trading_bot.active_tasks[user_id]

        safe_bot_reply(message, f"⏸️ Operações pausadas para você, {user_name}!\n\n✅ Use qualquer comando para continuar.")
        logger.info(f"Operações pausadas para usuário {user_name}")

    except Exception as e:
        logger.error(f"Erro no comando /pause: {str(e)}")
        safe_bot_reply(message, "❌ Erro ao pausar. Tente novamente.")

@bot.message_handler(commands=['restart'])
def restart_command(message):
    """Comando para reinicializar o bot sem parar o workflow"""
    try:
        user_name = message.from_user.first_name or "Usuário"
        user_id = message.from_user.id
        logger.info(f"Comando /restart recebido de {user_name} (ID: {user_id})")

        # Limpar estados do usuário
        if user_id in trading_bot.active_alerts:
            del trading_bot.active_alerts[user_id]
        if user_id in trading_bot.alert_states:
            del trading_bot.alert_states[user_id]
        if user_id in trading_bot.active_tasks:
            del trading_bot.active_tasks[user_id]
        trading_bot.paused_users.discard(user_id)
        trading_bot.processing_users.discard(user_id)

        # Limpar jobs do scheduler para este usuário
        schedule.clear(f'alert_user_{user_id}')

        safe_bot_reply(message, f"🔄 Bot reinicializado para você, {user_name}!\n\n✅ Estados limpos:\n• Alertas automáticos\n• Tarefas ativas\n• Cache de análises\n• Operações em andamento\n\n🚀 Pronto para novos comandos!")
        logger.info(f"Bot reinicializado para usuário {user_name}")

    except Exception as e:
        logger.error(f"Erro no comando /restart: {str(e)}")
        safe_bot_reply(message, "❌ Erro ao reinicializar. Tente novamente.")

@bot.message_handler(commands=['help'])
def help_command(message):
    try:
        logger.info(f"Comando /help recebido de {message.from_user.first_name}")

        help_message = """
                        🤖 AJUDA - OVECCHIA TRADING BOT

                        📋 COMANDOS DISPONÍVEIS:

                        📊 /analise [fonte] [estrategia] [ativo] [timeframe] [modelo] [data_inicio] [data_fim]
                          📝 ANÁLISE INDIVIDUAL COM GRÁFICO
                          • Gera gráfico completo do ativo escolhido
                          • Mostra sinais de compra/venda em tempo real
                          • Suporte a múltiplos timeframes e estratégias

                          🔗 Fontes: yahoo (padrão), ccxt, twelvedata
                          🎯 Estratégias: agressiva, balanceada, conservadora
                          🤖 Modelos: ovelha (padrão), ovelha2
                          ⏰ Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk
                          📅 Datas: YYYY-MM-DD

                          Exemplo básico: /analise yahoo balanceada PETR4.SA 1d
                          Com 12Data e ML: /analise twelvedata agressiva BTCUSDT 4h ovelha2

                        🔍 /screening [estrategia] [lista/ativos]
                          📝 SCREENING PONTUAL DE MÚLTIPLOS ATIVOS
                          • Verifica mudanças de estado em vários ativos
                          • Detecta oportunidades de compra/venda
                          • Análise instantânea de listas ou ativos individuais

                          Com lista: /screening balanceada açõesBR
                          Individual: /screening balanceada BTC-USD ETH-USD PETR4.SA
                          ⚠️ Configuração: Timeframe 1d fixo, 2 anos de dados

                        🔄 /screening_auto [fonte] [símbolos] [modelo] [estrategia] [timeframe]
                          📝 ALERTAS AUTOMÁTICOS DE SCREENING
                          • Monitora até 10 símbolos automaticamente
                          • Envia alertas quando detecta mudanças de estado
                          • Funciona no intervalo de tempo escolhido
                          • Suporte a múltiplas fontes de dados

                          🔗 Fontes: 12data, yahoo, ccxt
                          📊 Símbolos 12Data: [BTC/USD,ETH/USD,LTC/USD]
                          📊 Símbolos Yahoo: [BTC-USD,ETH-USD,PETR4.SA]
                          📊 Símbolos CCXT: [BTC/USDT,ETH/USDT,LTC/USDT]

                        ⏰ Timeframes: 5m (só 12Data), 15m, 1h, 4h, 1d

                        📋 /list_alerts
                          📝 VER ALERTAS ATIVOS
                          • Mostra configuração atual dos alertas
                          • Lista símbolos monitorados
                          • Exibe estratégia, modelo e timeframe configurados

                        🛑 /stop_alerts
                          📝 PARAR ALERTAS AUTOMÁTICOS
                          • Interrompe todos os alertas configurados
                          • Para o monitoramento automático

                        ⏸️ /pause
                          📝 PAUSAR OPERAÇÕES EM ANDAMENTO
                          • Cancela análises em processo
                          • Para tarefas que estão travando
                          • Use qualquer comando para continuar

                        🔄 /restart
                          📝 REINICIALIZAR BOT (sem parar o workflow)
                          • Limpa estados do usuário
                          • Resolve travamentos temporários
                          • Cancela tarefas ativas

                        ❓ /help - Esta mensagem de ajuda

                        🎯 ESTRATÉGIAS:
                        • agressiva - Mais sinais, maior frequência
                        • balanceada - Equilibrio entre sinais e confiabilidade (recomendada)
                        • conservadora - Sinais mais confiáveis, menor frequência

                        🤖 MODELOS:
                        • ovelha - Modelo clássico
                        • ovelha2 - Machine Learning (mais avançado)

                        📊 LISTAS PRÉ-DEFINIDAS:
                        • açõesBR - Ações brasileiras
                        • açõesEUA - Ações americanas
                        • criptos - Criptomoedas
                        • forex - Pares de moedas
                        • commodities - Commodities

                        ⏰ TIMEFRAMES POR COMANDO:
                        • /analise: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk
                        • /screening: 1d fixo
                        • /screening_auto: 5m, 15m, 1h, 4h, 1d (12Data apenas)

                        💡 EXEMPLOS PRÁTICOS:
                        • Análise rápida: /analise yahoo balanceada PETR4.SA 1d
                        • Análise cripto ML: /analise twelvedata agressiva BTCUSDT 4h ovelha2
                        • Screening geral: /screening balanceada açõesBR
                        • Alerta 12Data: /screening_auto [BTCUSDT,ETHUSDT] ovelha2 balanceada 4h

                        📝 FORMATOS DE SÍMBOLOS:
                        • Yahoo: PETR4.SA, AAPL, BTC-USD, EURUSD=X
                        • CCXT: BTC/USDT, ETH/USDT, LTC/USDT
                        • 12Data: BTCUSDT, ETHUSDT, EURUSD, AAPL

                        🔔 NOTA SOBRE 12DATA:
                        O comando /screening_auto agora usa exclusivamente 12Data e suporta timeframes a partir de 5 minutos, ideal para monitoramento de alta frequência de criptomoedas, forex e ações.
                        """
        safe_bot_reply(message, help_message)
    except Exception as e:
        logger.error(f"Erro no comando /help: {str(e)}")
        safe_bot_reply(message, "❌ Erro ao exibir ajuda.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_message = message.text or ""
        user_name = message.from_user.first_name or "Usuário"
        user_id = message.from_user.id
        chat_id = message.chat.id

        logger.info(f"📨 Mensagem de {user_name} (ID: {user_id}): {user_message}")
        print(f"📨 {user_name}: {user_message}")

        # Adicionar pequeno delay para evitar conflitos
        time.sleep(0.2)

        # Tentar identificar comando com fuzzy matching
        parsed = parse_flexible_command(user_message)

        if parsed:
            command = parsed['command']
            logger.info(f"Comando fuzzy identificado: {command} (original: {parsed['original_text']})")

            # Redirecionar para o handler apropriado
            if command == 'analise':
                analise_command(message)
            elif command == 'screening':
                screening_command(message)
            elif command == 'help':
                help_command(message)
            return

        # Mensagens de saudação
        user_message_lower = user_message.lower()
        if any(word in user_message_lower for word in ['oi', 'olá', 'hello', 'hi']):
            safe_bot_reply(message, "👋 Olá! Use /help para ver os comandos disponíveis.\n\n📊 Comandos principais:\n• /analise - Análise individual\n• /screening - Screening múltiplos ativos\n• /screening_auto - Alertas automáticos\n• /list_alerts - Ver alertas ativos\n• /stop_alerts - Parar alertas")
        elif any(word in user_message_lower for word in ['ajuda', 'help']):
            help_command(message)
        else:
            safe_bot_reply(message, "🤖 Use /help para ver os comandos disponíveis.\n\n📊 Comandos principais:\n• /analise - Análise individual\n• /screening - Screening múltiplos ativos\n• /screening_auto - Alertas automáticos (12Data)\n• /list_alerts - Ver alertas ativos\n• /stop_alerts - Parar alertas")

    except telebot.apihelper.ApiException as e:
        logger.error(f"Erro da API Telegram no handler de mensagem: {str(e)}")
    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {str(e)}")

def schedule_alerts_for_user(user_id, timeframe):
    """Programa alertas baseado no timeframe escolhido"""
    try:
        # Cancelar jobs existentes para este usuário
        schedule.clear(f'alert_user_{user_id}')

        # Programar nova tarefa baseada no timeframe
        if timeframe == '1m':
            schedule.every(1).minutes.do(send_scheduled_alert, user_id).tag(f'alert_user_{user_id}')
        elif timeframe == '5m':
            schedule.every(5).minutes.do(send_scheduled_alert, user_id).tag(f'alert_user_{user_id}')
        elif timeframe == '15m':
            schedule.every(15).minutes.do(send_scheduled_alert, user_id).tag(f'alert_user_{user_id}')
        elif timeframe == '1h':
            schedule.every(1).hours.do(send_scheduled_alert, user_id).tag(f'alert_user_{user_id}')
        elif timeframe == '4h':
            schedule.every(4).hours.do(send_scheduled_alert, user_id).tag(f'alert_user_{user_id}')
        elif timeframe == '1d':
            schedule.every(1).days.do(send_scheduled_alert, user_id).tag(f'alert_user_{user_id}')

        logger.info(f"Alerta programado para usuário {user_id} a cada {timeframe}")

    except Exception as e:
        logger.error(f"Erro ao programar alerta para usuário {user_id}: {str(e)}")

def send_scheduled_alert(user_id):
    """Envia alerta programado para um usuário específico"""
    try:
        if user_id not in trading_bot.active_alerts:
            logger.info(f"Alerta cancelado para usuário {user_id} - configuração removida")
            schedule.clear(f'alert_user_{user_id}')
            return

        alert_config = trading_bot.active_alerts[user_id]

        logger.info(f"Executando screening automático para usuário {user_id}")

        # Realizar screening
        current_states, changes = trading_bot.perform_automated_screening(
            user_id,
            alert_config['symbols'],
            alert_config['source'],
            alert_config['model'],
            alert_config['strategy'],
            alert_config['timeframe']
        )

        # Preparar mensagem
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

        if changes:
            # Mudanças detectadas
            message = f"🚨 *ALERTAS DE MUDANÇA DETECTADOS*\n📅 {timestamp}\n\n"
            message += f"⚙️ **Configuração:**\n"
            message += f"🔗 {alert_config['source'].upper()} | 🎯 {alert_config['strategy']} | 🤖 {alert_config['model'].upper()}\n"
            message += f"⏰ Intervalo: {alert_config['timeframe']}\n\n"

            for change in changes:
                prev_icon = "🔵" if change['previous_state'] == "Buy" else "🔴" if change['previous_state'] == "Sell" else "⚫"
                curr_icon = "🔵" if change['current_state'] == "Buy" else "🔴" if change['current_state'] == "Sell" else "⚫"

                message += f"📊 **{change['symbol']}**\n"
                message += f"💰 Preço: {change['current_price']:.4f}\n"
                message += f"🔄 {prev_icon} {change['previous_state']} → {curr_icon} {change['current_state']}\n\n"

            message += f"⏰ Próximo alerta em: {alert_config['timeframe']}"

        else:
            # Nenhuma mudança
            message = f"ℹ️ *SCREENING AUTOMÁTICO - SEM MUDANÇAS*\n📅 {timestamp}\n\n"
            message += f"⚙️ **Configuração:**\n"
            message += f"🔗 {alert_config['source'].upper()} | 🎯 {alert_config['strategy']} | 🤖 {alert_config['model'].upper()}\n"
            message += f"⏰ Intervalo: {alert_config['timeframe']}\n\n"

            message += f"📊 **Status Atual ({len(current_states)} símbolos):**\n"
            for symbol, state_info in current_states.items():
                state_icon = "🔵" if state_info['state'] == "Buy" else "🔴" if state_info['state'] == "Sell" else "⚫"
                message += f"• {symbol}: {state_icon} {state_info['state']} ({state_info['price']:.4f})\n"

            message += f"\n⏰ Próximo alerta em: {alert_config['timeframe']}"

        # Enviar mensagem
        bot.send_message(alert_config['chat_id'], message, parse_mode='Markdown')
        logger.info(f"Alerta enviado para usuário {user_id}: {len(changes)} mudanças detectadas")

    except Exception as e:
        logger.error(f"Erro ao enviar alerta programado para usuário {user_id}: {str(e)}")

def run_scheduler():
    """Thread separada para executar o scheduler"""
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)  # Verificar a cada 30 segundos (reduzido de 60)
        except Exception as e:
            logger.error(f"Erro no scheduler: {str(e)}")
            time.sleep(30)

def test_bot_connection():
    """Testa a conexão com a API do Telegram"""
    try:
        bot_info = bot.get_me()
        logger.info(f"✅ Conexão com Telegram OK - Bot: @{bot_info.username}")
        print(f"✅ Bot conectado: @{bot_info.username}")
        return True
    except Exception as e:
        logger.error(f"❌ Falha na conexão com Telegram: {str(e)}")
        print(f"❌ Falha na conexão: {str(e)}")
        return False

def run_bot():
    """Função para rodar o bot"""
    max_retries = 10  # Aumentado para mais tentativas
    retry_count = 0
    last_error_time = 0

    # Teste inicial de conectividade
    if not test_bot_connection():
        logger.error("❌ Não foi possível conectar ao Telegram. Verifique o token.")
        print("❌ Erro de conectividade. Bot não será iniciado.")
        return

    while retry_count < max_retries:
        try:
            logger.info("🤖 Iniciando OVECCHIA TRADING BOT...")
            print("🤖 OVECCHIA TRADING BOT ONLINE!")

            # Configurar comandos do bot
            try:
                bot.set_my_commands([
                    telebot.types.BotCommand("analise", "Análise individual com gráfico"),
                    telebot.types.BotCommand("screening", "Screening de múltiplos ativos"),
                    telebot.types.BotCommand("screening_auto", "Alertas automáticos de screening"),
                    telebot.types.BotCommand("list_alerts", "Ver alertas ativos"),
                    telebot.types.BotCommand("stop_alerts", "Parar alertas automáticos"),
                    telebot.types.BotCommand("help", "Ajuda com comandos")
                ])
                logger.info("✅ Comandos do bot configurados")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao configurar comandos: {str(e)}")

            # Iniciar thread do scheduler
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            logger.info("🔄 Scheduler de alertas iniciado")

            logger.info("🤖 Bot iniciado com sucesso! Aguardando mensagens...")
            print("🤖 Bot funcionando! Aguardando comandos...")

            # Rodar o bot com configurações otimizadas para maior estabilidade
            bot.polling(
                none_stop=True,
                interval=1,           # 1 segundo para evitar sobrecarga
                timeout=20,           # Timeout menor para evitar travamentos
                long_polling_timeout=20,  # Timeout do long polling menor
                allowed_updates=["message"],  # Apenas mensagens
                skip_pending=True     # Pular mensagens pendentes antigas
            )

        except telebot.apihelper.ApiException as e:
            current_time = time.time()
            logger.error(f"Erro da API do Telegram: {str(e)}")
            print(f"❌ Erro da API Telegram: {str(e)}")

            if "Unauthorized" in str(e) or "token" in str(e).lower():
                logger.error("❌ Token inválido ou expirado!")
                print("❌ ERRO CRÍTICO: Token do bot inválido!")
                break

            # Se o mesmo erro ocorreu recentemente, aumentar o tempo de espera
            if current_time - last_error_time < 60:  # Menos de 1 minuto desde o último erro
                retry_count += 2  # Penalizar mais por erros frequentes
            else:
                retry_count += 1

            last_error_time = current_time

            if retry_count < max_retries:
                wait_time = min(60, 5 * retry_count)  # Máximo 1 minuto de espera
                logger.info(f"🔄 Tentando novamente em {wait_time} segundos... (tentativa {retry_count}/{max_retries})")
                print(f"⏳ Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)

        except Exception as e:
            retry_count += 1
            logger.error(f"Erro crítico no bot (tentativa {retry_count}/{max_retries}): {str(e)}")
            print(f"❌ Erro ao iniciar bot (tentativa {retry_count}/{max_retries}): {str(e)}")

            # Limpar estados em caso de erro crítico
            trading_bot.active_alerts.clear()
            trading_bot.alert_states.clear()
            trading_bot.active_tasks.clear()
            trading_bot.paused_users.clear()
            schedule.clear()

            if retry_count < max_retries:
                wait_time = min(30, 5 * retry_count)  # Máximo 30s de espera
                logger.info(f"🔄 Estados limpos. Tentando novamente em {wait_time} segundos...")
                print(f"🧹 Limpando estados... Tentativa em {wait_time}s")
                time.sleep(wait_time)
            else:
                logger.error("🛑 Máximo de tentativas excedido. Bot será encerrado.")
                print("🛑 Bot será encerrado após múltiplas falhas.")
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