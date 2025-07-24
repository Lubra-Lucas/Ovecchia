
#!/usr/bin/env python3
import asyncio
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot token
BOT_TOKEN = "8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k"

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

# Initialize bot
bot = OvecchiaTradingBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /start"""
    try:
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name
        logger.info(f"Comando /start recebido de {user_name} (ID: {user_id})")
        
        welcome_message = """
🤖 *Bem-vindo ao OVECCHIA TRADING BOT!*

🎯 *Comandos disponíveis:*
/screening - Configurar screening de ativos
/topos_fundos - Detectar topos e fundos
/status - Ver status dos alertas
/help - Ajuda com comandos

📊 *Funcionalidades:*
• Screening automático de múltiplos ativos
• Detecção de topos e fundos
• Alertas em tempo real
• Timeframe: 1 dia
• Estratégias: Agressiva, Balanceada, Conservadora

🚀 Use /screening para começar!
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"Mensagem de boas-vindas enviada para {user_name}")
    except Exception as e:
        logger.error(f"Erro no comando /start: {str(e)}")
        await update.message.reply_text("❌ Erro interno. Tente novamente mais tarde.")

async def screening_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /screening"""
    try:
        user_name = update.effective_user.first_name
        logger.info(f"Comando /screening recebido de {user_name}")
        
        message = """
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

        # Se há argumentos, processar screening
        if context.args:
            await update.message.reply_text("🔄 Processando screening...", parse_mode='Markdown')
            
            strategy = "Balanceado"
            symbols = context.args

            # Verificar se o primeiro argumento é uma estratégia
            if context.args[0].lower() in ['agressiva', 'balanceada', 'conservadora']:
                strategy_map = {
                    'agressiva': 'Agressivo',
                    'balanceada': 'Balanceado',
                    'conservadora': 'Conservador'
                }
                strategy = strategy_map[context.args[0].lower()]
                symbols = context.args[1:]

            if not symbols:
                await update.message.reply_text("❌ Por favor, forneça pelo menos um ativo para análise.", parse_mode='Markdown')
                return

            logger.info(f"Realizando screening para {len(symbols)} ativos com estratégia {strategy}")

            # Realizar screening
            results = bot.perform_screening(symbols, strategy)

            if results:
                response = f"🚨 *ALERTAS DE MUDANÇA DE ESTADO*\n\n📊 Estratégia: {strategy}\n⏰ Timeframe: 1 dia\n\n"

                for result in results:
                    state_icon = "🟢" if result['current_state'] == "Buy" else "🔴" if result['current_state'] == "Sell" else "⚫"
                    prev_icon = "🟢" if result['previous_state'] == "Buy" else "🔴" if result['previous_state'] == "Sell" else "⚫"

                    response += f"{state_icon} *{result['symbol']}*\n"
                    response += f"💰 Preço: {result['current_price']:.2f}\n"
                    response += f"📈 {prev_icon} {result['previous_state']} → {state_icon} {result['current_state']}\n\n"

                await update.message.reply_text(response, parse_mode='Markdown')
                logger.info(f"Screening enviado para {user_name}: {len(results)} alertas")
            else:
                await update.message.reply_text("ℹ️ Nenhuma mudança de estado detectada nos ativos analisados.", parse_mode='Markdown')
                logger.info(f"Nenhum alerta encontrado para {user_name}")
        else:
            await update.message.reply_text(message, parse_mode='Markdown')
            
    except Exception as e:
        logger.error(f"Erro no comando /screening: {str(e)}")
        await update.message.reply_text("❌ Erro ao processar screening. Tente novamente.")

async def topos_fundos_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /topos_fundos"""
    try:
        user_name = update.effective_user.first_name
        logger.info(f"Comando /topos_fundos recebido de {user_name}")
        
        message = """
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

        if context.args:
            symbols = context.args
            await update.message.reply_text(f"🔄 Analisando topos e fundos para {len(symbols)} ativos...", parse_mode='Markdown')

            # Detectar topos e fundos
            results = bot.detect_tops_bottoms(symbols)

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

                await update.message.reply_text(response, parse_mode='Markdown')
                logger.info(f"Topos e fundos enviados para {user_name}: {len(results)} oportunidades")
            else:
                await update.message.reply_text("ℹ️ Nenhuma oportunidade de topo ou fundo detectada nos ativos analisados.", parse_mode='Markdown')
        else:
            await update.message.reply_text(message, parse_mode='Markdown')
            
    except Exception as e:
        logger.error(f"Erro no comando /topos_fundos: {str(e)}")
        await update.message.reply_text("❌ Erro ao processar topos e fundos. Tente novamente.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /status"""
    try:
        logger.info(f"Comando /status recebido de {update.effective_user.first_name}")
        
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
        await update.message.reply_text(status_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Erro no comando /status: {str(e)}")
        await update.message.reply_text("❌ Erro ao verificar status.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /help"""
    try:
        logger.info(f"Comando /help recebido de {update.effective_user.first_name}")
        
        help_message = """
🤖 *AJUDA - OVECCHIA TRADING BOT*

📋 *Comandos disponíveis:*

🏠 `/start` - Iniciar o bot

🔍 `/screening [estrategia] [ativos]`
   Exemplo: `/screening balanceada BTC-USD ETH-USD`

📊 `/topos_fundos [ativos]`
   Exemplo: `/topos_fundos PETR4.SA VALE3.SA`

📈 `/status` - Ver status do bot

❓ `/help` - Esta mensagem de ajuda

🎯 *Estratégias:*
• `agressiva` - Mais sinais
• `balanceada` - Equilibrada
• `conservadora` - Mais confiável

💡 *Exemplos de ativos:*
• Cripto: BTC-USD, ETH-USD, ADA-USD
• Ações BR: PETR4.SA, VALE3.SA, ITUB4.SA
• Ações US: AAPL, GOOGL, MSFT, TSLA
• Forex: EURUSD=X, GBPUSD=X
        """
        await update.message.reply_text(help_message, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Erro no comando /help: {str(e)}")
        await update.message.reply_text("❌ Erro ao exibir ajuda.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages"""
    try:
        user_message = update.message.text.lower()
        user_name = update.effective_user.first_name
        
        logger.info(f"Mensagem recebida de {user_name}: {user_message}")
        
        if any(word in user_message for word in ['oi', 'olá', 'hello', 'hi']):
            await update.message.reply_text("👋 Olá! Use /help para ver os comandos disponíveis.")
        elif 'ajuda' in user_message:
            await help_command(update, context)
        else:
            await update.message.reply_text("🤖 Use /help para ver os comandos disponíveis.")
            
    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {str(e)}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}")

async def main() -> None:
    """Main function to run the bot"""
    try:
        logger.info("🤖 Iniciando OVECCHIA TRADING BOT...")
        
        # Create application
        application = Application.builder().token(BOT_TOKEN).build()

        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("screening", screening_command))
        application.add_handler(CommandHandler("topos_fundos", topos_fundos_command))
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("help", help_command))
        
        # Add message handler for regular text
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Add error handler
        application.add_error_handler(error_handler)

        # Set bot commands for the menu
        commands = [
            BotCommand("start", "Iniciar o bot"),
            BotCommand("screening", "Screening de múltiplos ativos"),
            BotCommand("topos_fundos", "Detectar topos e fundos"),
            BotCommand("status", "Ver status do bot"),
            BotCommand("help", "Ajuda com comandos")
        ]

        await application.bot.set_my_commands(commands)
        
        logger.info("🤖 Bot iniciado com sucesso!")
        print("🤖 OVECCHIA TRADING BOT ONLINE!")
        
        # Run the bot
        await application.run_polling(
            timeout=10,
            drop_pending_updates=True,
            pool_timeout=30,
            connect_timeout=60,
            read_timeout=60,
            write_timeout=60
        )
        
    except Exception as e:
        logger.error(f"Erro crítico no bot: {str(e)}")
        print(f"❌ Erro ao iniciar bot: {str(e)}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot interrompido pelo usuário")
        print("🛑 Bot interrompido")
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        print(f"💥 Erro fatal: {str(e)}")
