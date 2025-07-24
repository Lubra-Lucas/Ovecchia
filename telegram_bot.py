
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
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot token
BOT_TOKEN = "8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k"

class OvecchiaTradingBot:
    def __init__(self):
        self.users_config = {}  # Store user configurations
        
    def get_market_data(self, symbol, start_date, end_date, interval):
        """Função para coletar dados do mercado"""
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
            logger.error(f"Erro ao coletar dados para {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_indicators_and_signals(self, df, strategy_type="Balanceado"):
        """Calcula indicadores e gera sinais"""
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
        results = []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        for symbol in symbols_list:
            try:
                df = self.get_market_data(symbol, start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"), "1d")
                
                if df.empty:
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
    user_id = update.effective_user.id
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

async def screening_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /screening"""
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
        strategy = "Balanceado"
        symbols = context.args
        
        # Verificar se o primeiro argumento é uma estratégia
        if context.args[0].lower() in ['agressiva', 'balanceada', 'conservadora']:
            strategy = context.args[0].capitalize() if context.args[0].lower() == 'agressiva' else context.args[0].capitalize()
            if context.args[0].lower() == 'agressiva':
                strategy = "Agressivo"
            elif context.args[0].lower() == 'conservadora':
                strategy = "Conservador"
            symbols = context.args[1:]
        
        if not symbols:
            await update.message.reply_text("❌ Por favor, forneça pelo menos um ativo para análise.", parse_mode='Markdown')
            return
        
        await update.message.reply_text(f"🔄 Analisando {len(symbols)} ativos com estratégia {strategy}...", parse_mode='Markdown')
        
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
        else:
            await update.message.reply_text("ℹ️ Nenhuma mudança de estado detectada nos ativos analisados.", parse_mode='Markdown')
    else:
        await update.message.reply_text(message, parse_mode='Markdown')

async def topos_fundos_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /topos_fundos"""
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
        else:
            await update.message.reply_text("ℹ️ Nenhuma oportunidade de topo ou fundo detectada nos ativos analisados.", parse_mode='Markdown')
    else:
        await update.message.reply_text(message, parse_mode='Markdown')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /status"""
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

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Comando /help"""
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

async def main() -> None:
    """Main function to run the bot"""
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("screening", screening_command))
    application.add_handler(CommandHandler("topos_fundos", topos_fundos_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Set bot commands for the menu
    commands = [
        BotCommand("start", "Iniciar o bot"),
        BotCommand("screening", "Screening de múltiplos ativos"),
        BotCommand("topos_fundos", "Detectar topos e fundos"),
        BotCommand("status", "Ver status do bot"),
        BotCommand("help", "Ajuda com comandos")
    ]
    
    await application.bot.set_my_commands(commands)
    
    # Run the bot
    print("🤖 Bot iniciado com sucesso!")
    await application.run_polling(timeout=10, drop_pending_updates=True)

if __name__ == '__main__':
    asyncio.run(main())
