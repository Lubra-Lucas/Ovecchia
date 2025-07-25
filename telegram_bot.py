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

class OvecchiaTradingBot:
    def __init__(self):
        self.users_config = {}

    def get_market_data(self, symbol, start_date, end_date, interval="1d"):
        """Função para coletar dados do mercado"""
        try:
            logger.info(f"Coletando dados para {symbol}")
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
        start_date = end_date - timedelta(days=365)

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
        start_date = end_date - timedelta(days=365)

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

    def generate_analysis_chart(self, symbol, strategy_type, timeframe, custom_start_date=None, custom_end_date=None):
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

            # Calcular indicadores e sinais
            df = self.calculate_indicators_and_signals(df, strategy_type)

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
            titulo_grafico = f"OVECCHIA TRADING - {symbol} - {timeframe.upper()}"
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
                caption = f"📊 OVECCHIA TRADING - {symbol}\n🎯 {strategy_type} | ⏰ {timeframe.upper()}\n📅 {custom_start_date} até {custom_end_date}"
            else:
                caption = f"📊 OVECCHIA TRADING - {symbol}\n🎯 {strategy_type} | ⏰ {timeframe.upper()}\n📅 Período: {start_date} até {end_date}"

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

        # Parse arguments
        args = message.text.split()[1:]  # Remove /screening from the list

        if not args:
            help_message = """
🔍 *SCREENING DE ATIVOS*

📝 *Como usar:*
/screening [estrategia] [ativos]

🎯 *Estratégias disponíveis:*
• agressiva - Mais sinais
• balanceada - Equilibrada (padrão)
• conservadora - Sinais mais confiáveis

📊 *Exemplo:*
`/screening balanceada BTC-USD ETH-USD PETR4.SA VALE3.SA`

💡 *Ativos suportados:*
• Cripto: BTC-USD, ETH-USD, etc.
• Ações BR: PETR4.SA, VALE3.SA, etc.
• Ações US: AAPL, GOOGL, etc.
            """
            bot.reply_to(message, help_message, parse_mode='Markdown')
            return

        bot.reply_to(message, "🔄 Processando screening...", parse_mode='Markdown')

        strategy = "Balanceado"
        symbols = args

        # Verificar se o primeiro argumento é uma estratégia
        if args[0].lower() in ['agressiva', 'balanceada', 'conservadora']:
            strategy_map = {
                'agressiva': 'Agressivo',
                'balanceada': 'Balanceado',
                'conservadora': 'Conservador'
            }
            strategy = strategy_map[args[0].lower()]
            symbols = args[1:]

        if not symbols:
            bot.reply_to(message, "❌ Por favor, forneça pelo menos um ativo para análise.", parse_mode='Markdown')
            return

        logger.info(f"Realizando screening para {len(symbols)} ativos com estratégia {strategy}")

        # Realizar screening
        results = trading_bot.perform_screening(symbols, strategy)

        if results:
            response = f"🚨 *ALERTAS DE MUDANÇA DE ESTADO*\n\n📊 Estratégia: {strategy}\n⏰ Timeframe: 1 dia\n\n"

            for result in results:
                state_icon = "🟢" if result['current_state'] == "Buy" else "🔴" if result['current_state'] == "Sell" else "⚫"
                prev_icon = "🟢" if result['previous_state'] == "Buy" else "🔴" if result['previous_state'] == "Sell" else "⚫"

                response += f"{state_icon} *{result['symbol']}*\n"
                response += f"💰 Preço: {result['current_price']:.2f}\n"
                response += f"📈 {prev_icon} {result['previous_state']} → {state_icon} {result['current_state']}\n\n"

            bot.reply_to(message, response, parse_mode='Markdown')
            logger.info(f"Screening enviado para {user_name}: {len(results)} alertas")
        else:
            bot.reply_to(message, "ℹ️ Nenhuma mudança de estado detectada nos ativos analisados.", parse_mode='Markdown')
            logger.info(f"Nenhum alerta encontrado para {user_name}")

    except Exception as e:
        logger.error(f"Erro no comando /screening: {str(e)}")
        bot.reply_to(message, "❌ Erro ao processar screening. Tente novamente.")

@bot.message_handler(commands=['topos_fundos'])
def topos_fundos_command(message):
    try:
        user_name = message.from_user.first_name
        logger.info(f"Comando /topos_fundos recebido de {user_name}")

        args = message.text.split()[1:]  # Remove /topos_fundos from the list

        if not args:
            help_message = """
📊 *DETECÇÃO DE TOPOS E FUNDOS*

📝 *Como usar:*
/topos_fundos [ativos]

📈 *Exemplo:*
`/topos_fundos BTC-USD ETH-USD PETR4.SA VALE3.SA AAPL GOOGL`

🎯 *O que detecta:*
• Possíveis fundos (oportunidades de compra)
• Possíveis topos (oportunidades de venda)
• Baseado em Bollinger Bands
• Timeframe: 1 dia
            """
            bot.reply_to(message, help_message, parse_mode='Markdown')
            return

        symbols = args
        bot.reply_to(message, f"🔄 Analisando topos e fundos para {len(symbols)} ativos...", parse_mode='Markdown')

        # Detectar topos e fundos
        results = trading_bot.detect_tops_bottoms(symbols)

        if results:
            response = "📊 *DETECÇÃO DE TOPOS E FUNDOS*\n\n⏰ Timeframe: 1 dia\n\n"

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

            bot.reply_to(message, response, parse_mode='Markdown')
            logger.info(f"Topos e fundos enviados para {user_name}: {len(results)} oportunidades")
        else:
            bot.reply_to(message, "ℹ️ Nenhuma oportunidade de topo ou fundo detectada nos ativos analisados.", parse_mode='Markdown')

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

        args = message.text.split()[1:]  # Remove /analise from the list

        if len(args) < 3:
            help_message = """📊 ANÁLISE INDIVIDUAL DE ATIVO

📝 Como usar:
/analise [estrategia] [ativo] [timeframe] [data_inicio] [data_fim]

🎯 Estratégias disponíveis:
• agressiva - Mais sinais, maior frequência
• balanceada - Equilibrada (recomendada)
• conservadora - Sinais mais confiáveis

⏰ Timeframes disponíveis:
1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk

📅 Formato de datas (opcional):
YYYY-MM-DD (exemplo: 2024-01-01)

📈 Exemplos:
/analise balanceada PETR4.SA 1d
/analise agressiva BTC-USD 4h 2024-01-01 2024-01-31
/analise conservadora AAPL 1d 2024-06-01 2024-12-01

💡 Ativos suportados:
• Cripto: BTC-USD, ETH-USD, etc.
• Ações BR: PETR4.SA, VALE3.SA, etc.
• Ações US: AAPL, GOOGL, etc.
• Forex: EURUSD=X, etc.

ℹ️ Se não especificar datas, será usado período padrão baseado no timeframe"""
            bot.reply_to(message, help_message)
            return

        strategy_input = args[0].lower()
        symbol = args[1].upper()
        timeframe = args[2].lower()

        # Datas opcionais
        start_date = None
        end_date = None

        if len(args) >= 5:
            try:
                start_date = args[3]
                end_date = args[4]
                # Validar formato de data
                datetime.strptime(start_date, '%Y-%m-%d')
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

        if start_date and end_date:
            bot.reply_to(message, f"🔄 Analisando {symbol} de {start_date} até {end_date} com estratégia {strategy_input} no timeframe {timeframe}...")
        else:
            bot.reply_to(message, f"🔄 Analisando {symbol} com estratégia {strategy_input} no timeframe {timeframe}...")

        # Gerar análise e gráfico
        chart_result = trading_bot.generate_analysis_chart(symbol, strategy, timeframe, start_date, end_date)

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

📊 /analise [estrategia] [ativo] [timeframe] [data_inicio] [data_fim]
   Exemplo: /analise balanceada PETR4.SA 1d
   Com datas: /analise balanceada PETR4.SA 1d 2024-01-01 2024-06-01

🔍 /screening [estrategia] [ativos]
   Exemplo: /screening balanceada BTC-USD ETH-USD

📈 /topos_fundos [ativos]
   Exemplo: /topos_fundos PETR4.SA VALE3.SA

📊 /status - Ver status do bot

🔄 /restart - Reiniciar o bot (em caso de problemas)

❓ /help - Esta mensagem de ajuda

🎯 ESTRATÉGIAS:
• agressiva - Mais sinais
• balanceada - Equilibrada
• conservadora - Mais confiável

⏰ TIMEFRAMES:
1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk

💡 EXEMPLOS DE ATIVOS:
• Cripto: BTC-USD, ETH-USD, ADA-USD
• Ações BR: PETR4.SA, VALE3.SA, ITUB4.SA
• Ações US: AAPL, GOOGL, MSFT, TSLA
• Forex: EURUSD=X, GBPUSD=X"""
        bot.reply_to(message, help_message)
    except Exception as e:
        logger.error(f"Erro no comando /help: {str(e)}")
        bot.reply_to(message, "❌ Erro ao exibir ajuda.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_message = message.text.lower()
        user_name = message.from_user.first_name

        logger.info(f"Mensagem recebida de {user_name}: {user_message}")

        if any(word in user_message for word in ['oi', 'olá', 'hello', 'hi']):
            bot.reply_to(message, "👋 Olá! Use /help para ver os comandos disponíveis.")
        elif 'ajuda' in user_message:
            help_command(message)
        else:
            bot.reply_to(message, "🤖 Use /help para ver os comandos disponíveis.")

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