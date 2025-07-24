
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

    def generate_analysis_chart(self, symbol, strategy_type, timeframe):
        """Gera gráfico de análise para um ativo específico"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            import tempfile
            import os
            
            # Configurar o engine de renderização para funcionar no Replit
            try:
                import kaleido
                pio.kaleido.scope.default_width = 1200
                pio.kaleido.scope.default_height = 600
            except ImportError:
                logger.warning("Kaleido não disponível, tentando alternativa")
                # Usar engine alternativo se Kaleido não estiver disponível
                pio.renderers.default = "svg"
            
            # Define período baseado no timeframe
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

            # Criar gráfico
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
                subplot_titles=(f"{symbol} - {strategy_type} - {timeframe.upper()}", "Indicador de Sinais")
            )

            # Adicionar linha de preço com cores dos sinais
            for i in range(len(df) - 1):
                color = 'blue' if df['Estado'].iloc[i] == 'Buy' else 'red' if df['Estado'].iloc[i] == 'Sell' else 'gray'
                fig.add_trace(go.Scatter(
                    x=df['time'][i:i+2],
                    y=df['close'][i:i+2],
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo="skip"
                ), row=1, col=1)

            # Adicionar médias móveis se disponíveis
            strategy_params = {
                "Agressivo": (10, 21),
                "Conservador": (140, 200),
                "Balanceado": (60, 70)
            }
            
            sma_short, sma_long = strategy_params.get(strategy_type, (60, 70))
            
            if f'SMA_{sma_short}' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df[f'SMA_{sma_short}'],
                    mode="lines", name=f'SMA {sma_short}',
                    line=dict(color="orange", width=1, dash="dot")
                ), row=1, col=1)
                
            if f'SMA_{sma_long}' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df[f'SMA_{sma_long}'],
                    mode="lines", name=f'SMA {sma_long}',
                    line=dict(color="purple", width=1, dash="dot")
                ), row=1, col=1)

            # Adicionar indicador de sinais
            indicator_mapping = {'Buy': 1, 'Sell': 0, 'Stay Out': 0.5}
            df['Indicator'] = df['Estado'].map(indicator_mapping)
            
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['Indicator'],
                mode="lines+markers",
                name="Signal",
                line=dict(color="purple", width=2),
                marker=dict(size=4),
                showlegend=False
            ), row=2, col=1)

            # Atualizar layout
            fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 0.5, 1], 
                           ticktext=['Venda', 'Neutro', 'Compra'], row=2, col=1)
            
            fig.update_layout(
                title=f"OVECCHIA TRADING - {symbol}",
                template="plotly_white",
                height=600,
                showlegend=True,
                font=dict(size=10)
            )

            # Salvar gráfico temporariamente
            temp_dir = tempfile.gettempdir()
            chart_filename = f"chart_{symbol}_{int(datetime.now().timestamp())}.png"
            chart_path = os.path.join(temp_dir, chart_filename)
            
            # Tentar salvar a imagem com diferentes métodos
            try:
                fig.write_image(chart_path, width=1200, height=600, scale=2, engine="kaleido")
            except Exception as img_error:
                logger.warning(f"Erro com Kaleido: {img_error}")
                try:
                    # Tentar com engine alternativo
                    fig.write_image(chart_path, width=1200, height=600, scale=1, engine="auto")
                except Exception as img_error2:
                    logger.error(f"Erro ao salvar imagem: {img_error2}")
                    # Como último recurso, salvar como HTML e informar o usuário
                    html_path = chart_path.replace('.png', '.html')
                    fig.write_html(html_path)
                    return {
                        'success': False, 
                        'error': 'Erro na geração de imagem. Tente novamente em alguns minutos.'
                    }

            # Preparar informações atuais
            current_price = df['close'].iloc[-1]
            current_state = df['Estado'].iloc[-1]
            current_rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 'N/A'
            
            # Estado anterior para detectar mudanças
            previous_state = df['Estado'].iloc[-2] if len(df) > 1 else current_state
            state_change = "✅ MUDANÇA" if current_state != previous_state else "➖ SEM MUDANÇA"

            caption = f"""📊 <b>ANÁLISE - {symbol}</b>

🎯 <b>Estratégia:</b> {strategy_type}
⏰ <b>Timeframe:</b> {timeframe.upper()}
💰 <b>Preço Atual:</b> {current_price:.2f}

📈 <b>Estado Atual:</b> {current_state}
📊 <b>Estado Anterior:</b> {previous_state}
🔄 <b>Mudança:</b> {state_change}

📉 <b>RSI (14):</b> {current_rsi if isinstance(current_rsi, str) else f"{current_rsi:.2f}"}

🤖 <i>OVECCHIA TRADING BOT</i>"""

            return {
                'success': True,
                'chart_path': chart_path,
                'caption': caption
            }

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico para {symbol}: {str(e)}")
            
            # Se falhar na geração de gráfico, retornar análise em texto
            try:
                # Ainda tentar coletar os dados para análise textual
                if timeframe in ['1m', '5m', '15m', '30m']:
                    days = 7
                elif timeframe in ['1h', '4h']:
                    days = 30
                else:
                    days = 180
                    
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days)
                
                df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"), timeframe)
                
                if not df.empty:
                    df = self.calculate_indicators_and_signals(df, strategy_type)
                    
                    current_price = df['close'].iloc[-1]
                    current_state = df['Estado'].iloc[-1]
                    current_rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 'N/A'
                    previous_state = df['Estado'].iloc[-2] if len(df) > 1 else current_state
                    state_change = "✅ MUDANÇA" if current_state != previous_state else "➖ SEM MUDANÇA"
                    
                    text_analysis = f"""📊 <b>ANÁLISE - {symbol}</b> (Texto)

🎯 <b>Estratégia:</b> {strategy_type}
⏰ <b>Timeframe:</b> {timeframe.upper()}
💰 <b>Preço Atual:</b> {current_price:.2f}

📈 <b>Estado Atual:</b> {current_state}
📊 <b>Estado Anterior:</b> {previous_state}
🔄 <b>Mudança:</b> {state_change}

📉 <b>RSI (14):</b> {current_rsi if isinstance(current_rsi, str) else f"{current_rsi:.2f}"}

⚠️ <i>Gráfico indisponível temporariamente</i>
🤖 <i>OVECCHIA TRADING BOT</i>"""
                    
                    return {
                        'success': True,
                        'chart_path': None,
                        'caption': text_analysis,
                        'text_only': True
                    }
            except:
                pass
                
            return {'success': False, 'error': f'Erro ao gerar análise: Sistema temporariamente indisponível'}

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
/analise [estrategia] [ativo] [timeframe]

🎯 Estratégias disponíveis:
• agressiva - Mais sinais, maior frequência
• balanceada - Equilibrada (recomendada)
• conservadora - Sinais mais confiáveis

⏰ Timeframes disponíveis:
1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk

📈 Exemplos:
/analise balanceada PETR4.SA 1d
/analise agressiva BTC-USD 4h
/analise conservadora AAPL 1d

💡 Ativos suportados:
• Cripto: BTC-USD, ETH-USD, etc.
• Ações BR: PETR4.SA, VALE3.SA, etc.
• Ações US: AAPL, GOOGL, etc.
• Forex: EURUSD=X, etc."""
            bot.reply_to(message, help_message)
            return

        strategy_input = args[0].lower()
        symbol = args[1].upper()
        timeframe = args[2].lower()

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

        bot.reply_to(message, f"🔄 Analisando {symbol} com estratégia {strategy_input} no timeframe {timeframe}...")
        
        # Gerar análise e gráfico
        chart_result = trading_bot.generate_analysis_chart(symbol, strategy, timeframe)
        
        if chart_result['success']:
            if chart_result.get('text_only', False):
                # Enviar apenas texto quando gráfico não disponível
                bot.reply_to(message, chart_result['caption'], parse_mode='HTML')
            else:
                # Enviar gráfico normal
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

@bot.message_handler(commands=['help'])
def help_command(message):
    try:
        logger.info(f"Comando /help recebido de {message.from_user.first_name}")
        
        help_message = """🤖 AJUDA - OVECCHIA TRADING BOT

📋 COMANDOS DISPONÍVEIS:

🏠 /start - Iniciar o bot

📊 /analise [estrategia] [ativo] [timeframe]
   Exemplo: /analise balanceada PETR4.SA 1d

🔍 /screening [estrategia] [ativos]
   Exemplo: /screening balanceada BTC-USD ETH-USD

📈 /topos_fundos [ativos]
   Exemplo: /topos_fundos PETR4.SA VALE3.SA

📊 /status - Ver status do bot

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
            telebot.types.BotCommand("help", "Ajuda com comandos")
        ])
        
        logger.info("🤖 Bot iniciado com sucesso!")
        
        # Rodar o bot
        bot.polling(none_stop=True, interval=2, timeout=30)
        
    except Exception as e:
        logger.error(f"Erro crítico no bot: {str(e)}")
        print(f"❌ Erro ao iniciar bot: {str(e)}")
        time.sleep(5)  # Aguardar antes de tentar novamente

if __name__ == '__main__':
    try:
        run_bot()
    except KeyboardInterrupt:
        logger.info("Bot interrompido pelo usuário")
        print("🛑 Bot interrompido")
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        print(f"💥 Erro fatal: {str(e)}")
