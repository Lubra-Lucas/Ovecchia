
#!/usr/bin/env python3
import asyncio
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO,
    handlers=[
        logging.FileHandler('telegram_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Bot token
BOT_TOKEN = "8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k"

class OvecchiaTradingBot:
    def __init__(self):
        self.users_config = {}
        logger.info("🤖 Inicializando Ovecchia Trading Bot...")
        
    def get_market_data(self, symbol, start_date, end_date, interval):
        """Função para coletar dados do mercado"""
        try:
            logger.info(f"📊 Coletando dados para {symbol}")
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            
            if df is None or df.empty:
                logger.warning(f"⚠️ Nenhum dado encontrado para {symbol}")
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
            
            logger.info(f"✅ Dados coletados com sucesso para {symbol}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao coletar dados para {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_indicators_and_signals(self, df, strategy_type="Balanceado"):
        """Calcula indicadores e gera sinais"""
        if df.empty:
            return df
            
        logger.info(f"🔢 Calculando indicadores com estratégia: {strategy_type}")
            
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
    
    def perform_screening(self, symbols_list, strategy_type="Balanceado"):
        """Realiza screening de múltiplos ativos"""
        logger.info(f"🔍 Iniciando screening de {len(symbols_list)} ativos")
        results = []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        for symbol in symbols_list:
            try:
                df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"), "1d")
                
                if df.empty:
                    logger.warning(f"⚠️ Dados vazios para {symbol}")
                    continue
                
                df = self.calculate_indicators_and_signals(df, strategy_type)
                
                if len(df) > 1:
                    current_state = df['Estado'].iloc[-1]
                    previous_state = df['Estado'].iloc[-2]
                    
                    if current_state != previous_state:
                        logger.info(f"🚨 Mudança de estado detectada para {symbol}: {previous_state} → {current_state}")
                        results.append({
                            'symbol': symbol,
                            'current_state': current_state,
                            'previous_state': previous_state,
                            'current_price': df['close'].iloc[-1]
                        })
                        
            except Exception as e:
                logger.error(f"❌ Erro ao analisar {symbol}: {str(e)}")
                continue
        
        logger.info(f"✅ Screening concluído. {len(results)} mudanças de estado encontradas")
        return results
    
    def detect_tops_bottoms(self, symbols_list):
        """Detecta topos e fundos usando Bollinger Bands"""
        logger.info(f"📊 Iniciando detecção de topos e fundos para {len(symbols_list)} ativos")
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
                    logger.info(f"🟢 Possível fundo detectado para {symbol}: {distance_pct:.2f}%")
                elif current_price > current_banda_superior:
                    distance_pct = ((current_price - current_banda_superior) / current_price) * 100
                    signal = 'Possível Topo (Venda)'
                    logger.info(f"🔴 Possível topo detectado para {symbol}: {distance_pct:.2f}%")
                
                if signal:
                    results.append({
                        'symbol': symbol,
                        'signal': signal,
                        'current_price': current_price,
                        'distance_pct': distance_pct
                    })
                    
            except Exception as e:
                logger.error(f"❌ Erro ao analisar topos/fundos {symbol}: {str(e)}")
                continue
        
        logger.info(f"✅ Detecção concluída. {len(results)} oportunidades encontradas")
        return results

# Initialize bot
bot = OvecchiaTradingBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /start"""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    logger.info(f"👤 Usuário {user_name} (ID: {user_id}) iniciou o bot")
    
    welcome_message = f"""
🤖 *Olá {user_name}! Bem-vindo ao OVECCHIA TRADING BOT!*

🎯 *Comandos disponíveis:*
/screening - Configurar screening de ativos
/topos\_fundos - Detectar topos e fundos
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
    await update.message.reply_text(welcome_message, parse_mode='MarkdownV2')

async def screening_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /screening"""
    user_id = update.effective_user.id
    logger.info(f"🔍 Usuário {user_id} executou comando /screening com args: {context.args}")
    
    message = """
🔍 *SCREENING DE ATIVOS*

📝 *Como usar:*
/screening \[estrategia\] \[ativos\]

🎯 *Estratégias disponíveis:*
• agressiva \- Mais sinais
• balanceada \- Equilibrada \(padrão\)
• conservadora \- Sinais mais confiáveis

📊 *Exemplo:*
`/screening balanceada BTC\-USD ETH\-USD PETR4\.SA VALE3\.SA`

💡 *Ativos suportados:*
• Cripto: BTC\-USD, ETH\-USD, etc\.
• Ações BR: PETR4\.SA, VALE3\.SA, etc\.
• Ações US: AAPL, GOOGL, etc\.
    """
    
    # Se há argumentos, processar screening
    if context.args:
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
            await update.message.reply_text("❌ Por favor, forneça pelo menos um ativo para análise\.", parse_mode='MarkdownV2')
            return
        
        await update.message.reply_text(f"🔄 Analisando {len(symbols)} ativos com estratégia {strategy}\.\.\.", parse_mode='MarkdownV2')
        
        # Realizar screening
        try:
            results = bot.perform_screening(symbols, strategy)
            
            if results:
                response = f"🚨 *ALERTAS DE MUDANÇA DE ESTADO*\n\n📊 Estratégia: {strategy}\n⏰ Timeframe: 1 dia\n\n"
                
                for result in results:
                    state_icon = "🟢" if result['current_state'] == "Buy" else "🔴" if result['current_state'] == "Sell" else "⚫"
                    prev_icon = "🟢" if result['previous_state'] == "Buy" else "🔴" if result['previous_state'] == "Sell" else "⚫"
                    
                    # Escape special characters for MarkdownV2
                    symbol_escaped = result['symbol'].replace('-', '\\-').replace('.', '\\.')
                    price_str = f"{result['current_price']:.2f}".replace('.', '\\.')
                    
                    response += f"{state_icon} *{symbol_escaped}*\n"
                    response += f"💰 Preço: {price_str}\n"
                    response += f"📈 {prev_icon} {result['previous_state']} → {state_icon} {result['current_state']}\n\n"
                
                await update.message.reply_text(response, parse_mode='MarkdownV2')
            else:
                await update.message.reply_text("ℹ️ Nenhuma mudança de estado detectada nos ativos analisados\\.", parse_mode='MarkdownV2')
                
        except Exception as e:
            logger.error(f"❌ Erro no screening: {str(e)}")
            await update.message.reply_text("❌ Erro ao realizar screening\\. Tente novamente\\.", parse_mode='MarkdownV2')
    else:
        await update.message.reply_text(message, parse_mode='MarkdownV2')

async def topos_fundos_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /topos_fundos"""
    user_id = update.effective_user.id
    logger.info(f"📊 Usuário {user_id} executou comando /topos_fundos com args: {context.args}")
    
    message = """
📊 *DETECÇÃO DE TOPOS E FUNDOS*

📝 *Como usar:*
/topos\_fundos \[ativos\]

📈 *Exemplo:*
`/topos\_fundos BTC\-USD ETH\-USD PETR4\.SA VALE3\.SA AAPL GOOGL`

🎯 *O que detecta:*
• Possíveis fundos \(oportunidades de compra\)
• Possíveis topos \(oportunidades de venda\)
• Baseado em Bollinger Bands
• Timeframe: 1 dia
    """
    
    if context.args:
        symbols = context.args
        
        await update.message.reply_text(f"🔄 Analisando topos e fundos para {len(symbols)} ativos\.\.\.", parse_mode='MarkdownV2')
        
        # Detectar topos e fundos
        try:
            results = bot.detect_tops_bottoms(symbols)
            
            if results:
                response = "📊 *DETECÇÃO DE TOPOS E FUNDOS*\n\n⏰ Timeframe: 1 dia\n\n"
                
                buy_opportunities = [r for r in results if 'Compra' in r['signal']]
                sell_opportunities = [r for r in results if 'Venda' in r['signal']]
                
                if buy_opportunities:
                    response += "🟢 *POSSÍVEIS FUNDOS \\(COMPRA\\):*\n"
                    for result in buy_opportunities:
                        symbol_escaped = result['symbol'].replace('-', '\\-').replace('.', '\\.')
                        price_str = f"{result['current_price']:.2f}".replace('.', '\\.')
                        distance_str = f"{result['distance_pct']:.2f}".replace('.', '\\.')
                        
                        response += f"• *{symbol_escaped}*: {price_str}\n"
                        response += f"  📊 Distância: {distance_str}%\n\n"
                
                if sell_opportunities:
                    response += "🔴 *POSSÍVEIS TOPOS \\(VENDA\\):*\n"
                    for result in sell_opportunities:
                        symbol_escaped = result['symbol'].replace('-', '\\-').replace('.', '\\.')
                        price_str = f"{result['current_price']:.2f}".replace('.', '\\.')
                        distance_str = f"{result['distance_pct']:.2f}".replace('.', '\\.')
                        
                        response += f"• *{symbol_escaped}*: {price_str}\n"
                        response += f"  📊 Distância: {distance_str}%\n\n"
                
                await update.message.reply_text(response, parse_mode='MarkdownV2')
            else:
                await update.message.reply_text("ℹ️ Nenhuma oportunidade de topo ou fundo detectada nos ativos analisados\\.", parse_mode='MarkdownV2')
                
        except Exception as e:
            logger.error(f"❌ Erro na detecção de topos/fundos: {str(e)}")
            await update.message.reply_text("❌ Erro ao detectar topos e fundos\\. Tente novamente\\.", parse_mode='MarkdownV2')
    else:
        await update.message.reply_text(message, parse_mode='MarkdownV2')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /status"""
    user_id = update.effective_user.id
    logger.info(f"📈 Usuário {user_id} executou comando /status")
    
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    status_message = f"""
📊 *STATUS DO BOT*

🤖 Bot: Online ✅
⏰ Timeframe: 1 dia
📅 Período análise: 365 dias
🔄 Última verificação: {current_time}

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

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /help"""
    user_id = update.effective_user.id
    logger.info(f"❓ Usuário {user_id} executou comando /help")
    
    help_message = """
🤖 *AJUDA \\- OVECCHIA TRADING BOT*

📋 *Comandos disponíveis:*

🏠 `/start` \\- Iniciar o bot

🔍 `/screening [estrategia] [ativos]`
   Exemplo: `/screening balanceada BTC\\-USD ETH\\-USD`

📊 `/topos_fundos [ativos]`
   Exemplo: `/topos_fundos PETR4\\.SA VALE3\\.SA`

📈 `/status` \\- Ver status do bot

❓ `/help` \\- Esta mensagem de ajuda

🎯 *Estratégias:*
• `agressiva` \\- Mais sinais
• `balanceada` \\- Equilibrada
• `conservadora` \\- Mais confiável

💡 *Exemplos de ativos:*
• Cripto: BTC\\-USD, ETH\\-USD, ADA\\-USD
• Ações BR: PETR4\\.SA, VALE3\\.SA, ITUB4\\.SA
• Ações US: AAPL, GOOGL, MSFT, TSLA
• Forex: EURUSD=X, GBPUSD=X
    """
    await update.message.reply_text(help_message, parse_mode='MarkdownV2')

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    logger.error(f"❌ Exception while handling an update: {context.error}")

async def main() -> None:
    """Main function to run the bot"""
    try:
        logger.info("🚀 Iniciando Ovecchia Trading Bot...")
        
        # Create application
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("screening", screening_command))
        application.add_handler(CommandHandler("topos_fundos", topos_fundos_command))
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("help", help_command))
        
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
        logger.info("✅ Comandos do bot configurados")
        
        # Test bot connection
        me = await application.bot.get_me()
        logger.info(f"✅ Bot conectado com sucesso: @{me.username}")
        
        # Run the bot
        logger.info("🔄 Iniciando polling...")
        await application.run_polling(
            timeout=30,
            drop_pending_updates=True,
            close_loop=False
        )
        
    except Exception as e:
        logger.error(f"❌ Erro crítico no bot: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Create a new event loop for the bot
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {str(e)}")
    finally:
        logger.info("🔚 Bot finalizado")
