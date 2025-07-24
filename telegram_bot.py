
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
import json
import os

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')  # Get from environment variable
USERS_FILE = 'telegram_users.json'

# User data storage
def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# Market data functions (imported from main app)
def get_market_data(symbol, start_date, end_date, interval):
    """Get market data using Yahoo Finance"""
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        
        if df is None or df.empty:
            return pd.DataFrame()
            
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df = df.xs(symbol, level='Ticker', axis=1, drop_level=True)
            
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
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def analyze_symbol(symbol, strategy="Balanceado"):
    """Analyze a single symbol for state changes"""
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        df = get_market_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), "1d")
        
        if df.empty:
            return None
            
        # Set strategy parameters
        if strategy == "Agressivo":
            sma_short, sma_long = 10, 21
        elif strategy == "Conservador":
            sma_short, sma_long = 140, 200
        else:  # Balanceado
            sma_short, sma_long = 60, 70
            
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
            
            if (df['close'].iloc[i] > df[f'SMA_{sma_short}'].iloc[i] and 
                df['close'].iloc[i] > df[f'SMA_{sma_long}'].iloc[i] and 
                rsi_up and rsl_buy):
                df.at[i, 'Signal'] = 'Buy'
            elif (df['close'].iloc[i] < df[f'SMA_{sma_short}'].iloc[i] and 
                  rsi_down and rsl_sell):
                df.at[i, 'Signal'] = 'Sell'
        
        # State persistence
        df['Estado'] = 'Stay Out'
        for i in range(len(df)):
            if i == 0:
                continue
            estado_anterior = df['Estado'].iloc[i-1]
            sinal_atual = df['Signal'].iloc[i]
            if sinal_atual != 'Stay Out':
                df.loc[df.index[i], 'Estado'] = sinal_atual
            else:
                df.loc[df.index[i], 'Estado'] = estado_anterior
        
        # Check for state change
        current_state = df['Estado'].iloc[-1]
        previous_state = df['Estado'].iloc[-2] if len(df) > 1 else current_state
        current_price = df['close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'current_state': current_state,
            'previous_state': previous_state,
            'state_change': current_state != previous_state,
            'current_price': current_price
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

def analyze_bollinger_bands(symbol):
    """Analyze symbol for tops and bottoms using Bollinger Bands"""
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        df = get_market_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), "1d")
        
        if df.empty:
            return None
            
        # Bollinger Bands calculation
        period = 20
        std_dev = 2.0
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        banda_superior = sma + (std_dev * std)
        banda_inferior = sma - (std_dev * std)
        
        current_price = df['close'].iloc[-1]
        current_banda_superior = banda_superior.iloc[-1]
        current_banda_inferior = banda_inferior.iloc[-1]
        
        signal = 'Neutro'
        if current_price < current_banda_inferior:
            signal = 'Possível Fundo (Compra)'
        elif current_price > current_banda_superior:
            signal = 'Possível Topo (Venda)'
            
        return {
            'symbol': symbol,
            'signal': signal,
            'current_price': current_price,
            'banda_superior': current_banda_superior,
            'banda_inferior': current_banda_inferior
        }
        
    except Exception as e:
        logger.error(f"Error analyzing Bollinger Bands for {symbol}: {e}")
        return None

# Bot command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    keyboard = [
        [InlineKeyboardButton("🔄 Verificar Mudança de Estado", callback_data='screening')],
        [InlineKeyboardButton("📊 Verificar Topos e Fundos", callback_data='topobottom')],
        [InlineKeyboardButton("🎯 Ambos", callback_data='both')],
        [InlineKeyboardButton("⚙️ Configurações", callback_data='settings')],
        [InlineKeyboardButton("📋 Status", callback_data='status')]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = """
🤖 **Bem-vindo ao OVECCHIA TRADING Bot!**

Receba alertas automáticos sobre:
• 🔄 Mudanças de estado dos ativos
• 📊 Detecção de topos e fundos
• 🎯 Oportunidades de trading

O que você gostaria de fazer?
    """
    
    await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    users = load_users()
    user_id = str(query.from_user.id)
    
    if user_id not in users:
        users[user_id] = {
            'username': query.from_user.username,
            'alerts_enabled': False,
            'strategy': 'Balanceado',
            'symbols': [],
            'alert_types': []
        }
    
    if query.data == 'screening':
        keyboard = [
            [InlineKeyboardButton("🔥 Agressivo", callback_data='strategy_agressivo')],
            [InlineKeyboardButton("⚖️ Balanceado", callback_data='strategy_balanceado')],
            [InlineKeyboardButton("🛡️ Conservador", callback_data='strategy_conservador')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📈 **Screening de Mudanças de Estado**\n\nEscolha sua estratégia:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    elif query.data == 'topobottom':
        users[user_id]['alert_types'] = ['topos_fundos']
        save_users(users)
        
        await query.edit_message_text(
            "📊 **Detecção de Topos e Fundos ativada!**\n\nAgora envie os tickers que deseja monitorar (exemplo: BTC-USD, PETR4.SA, AAPL)"
        )
        
    elif query.data == 'both':
        keyboard = [
            [InlineKeyboardButton("🔥 Agressivo", callback_data='strategy_both_agressivo')],
            [InlineKeyboardButton("⚖️ Balanceado", callback_data='strategy_both_balanceado')],
            [InlineKeyboardButton("🛡️ Conservador", callback_data='strategy_both_conservador')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "🎯 **Ambos os Alertas**\n\nEscolha sua estratégia para screening:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    elif query.data.startswith('strategy_'):
        strategy_map = {
            'agressivo': 'Agressivo',
            'balanceado': 'Balanceado', 
            'conservador': 'Conservador'
        }
        
        if 'both' in query.data:
            strategy = strategy_map[query.data.split('_')[2]]
            users[user_id]['alert_types'] = ['mudanca_estado', 'topos_fundos']
        else:
            strategy = strategy_map[query.data.split('_')[1]]
            users[user_id]['alert_types'] = ['mudanca_estado']
            
        users[user_id]['strategy'] = strategy
        save_users(users)
        
        await query.edit_message_text(
            f"✅ **Estratégia {strategy} selecionada!**\n\nAgora envie os tickers que deseja monitorar (exemplo: BTC-USD, PETR4.SA, AAPL)"
        )
        
    elif query.data == 'status':
        user_data = users.get(user_id, {})
        symbols = user_data.get('symbols', [])
        strategy = user_data.get('strategy', 'Não configurado')
        alert_types = user_data.get('alert_types', [])
        alerts_enabled = user_data.get('alerts_enabled', False)
        
        status_text = f"""
📊 **Status dos seus Alertas**

🎯 **Estratégia:** {strategy}
📈 **Ativos Monitorados:** {len(symbols)}
🔔 **Alertas Ativos:** {'✅ Sim' if alerts_enabled else '❌ Não'}

**Tipos de Alerta:**
{('• Mudanças de Estado\\n' if 'mudanca_estado' in alert_types else '') + 
 ('• Topos e Fundos\\n' if 'topos_fundos' in alert_types else '')}

**Ativos:** {', '.join(symbols) if symbols else 'Nenhum configurado'}
        """
        
        keyboard = [[InlineKeyboardButton("🔄 Reconfigurar", callback_data='start')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(status_text, reply_markup=reply_markup, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    users = load_users()
    user_id = str(update.effective_user.id)
    
    if user_id not in users:
        await update.message.reply_text("Use /start para começar!")
        return
    
    text = update.message.text.upper()
    symbols = [s.strip() for s in text.replace(',', ' ').split() if s.strip()]
    
    # Validate symbols
    valid_symbols = []
    for symbol in symbols:
        try:
            # Quick validation by trying to fetch 1 day of data
            test_df = get_market_data(symbol, (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), 
                                    datetime.now().strftime("%Y-%m-%d"), "1d")
            if not test_df.empty:
                valid_symbols.append(symbol)
        except:
            continue
    
    if valid_symbols:
        users[user_id]['symbols'] = valid_symbols
        users[user_id]['alerts_enabled'] = True
        save_users(users)
        
        await update.message.reply_text(
            f"✅ **Alertas Configurados!**\n\n"
            f"📈 **Ativos:** {', '.join(valid_symbols)}\n"
            f"🎯 **Estratégia:** {users[user_id].get('strategy', 'Balanceado')}\n"
            f"🔔 **Tipos:** {', '.join(users[user_id].get('alert_types', []))}\n\n"
            f"Você receberá alertas automáticos sobre estes ativos!"
        )
    else:
        await update.message.reply_text(
            "❌ **Nenhum ticker válido encontrado.**\n\n"
            "Certifique-se de usar o formato correto:\n"
            "• Criptomoedas: BTC-USD, ETH-USD\n" 
            "• Ações BR: PETR4.SA, VALE3.SA\n"
            "• Ações US: AAPL, GOOGL\n"
            "• Forex: EURUSD=X"
        )

async def send_alerts():
    """Check for alerts and send to users"""
    users = load_users()
    
    for user_id, user_data in users.items():
        if not user_data.get('alerts_enabled', False):
            continue
            
        symbols = user_data.get('symbols', [])
        strategy = user_data.get('strategy', 'Balanceado')
        alert_types = user_data.get('alert_types', [])
        
        alerts_to_send = []
        
        for symbol in symbols:
            # Check for state changes
            if 'mudanca_estado' in alert_types:
                result = analyze_symbol(symbol, strategy)
                if result and result['state_change']:
                    state_icon = "🔵" if result['current_state'] == "Buy" else "🔴" if result['current_state'] == "Sell" else "⚫"
                    alerts_to_send.append(
                        f"🔄 **{symbol}**\n"
                        f"{state_icon} **{result['current_state']}**\n"
                        f"💰 Preço: {result['current_price']:.2f}"
                    )
            
            # Check for tops and bottoms
            if 'topos_fundos' in alert_types:
                bb_result = analyze_bollinger_bands(symbol)
                if bb_result and bb_result['signal'] != 'Neutro':
                    signal_icon = "🟢" if "Compra" in bb_result['signal'] else "🔴"
                    alerts_to_send.append(
                        f"📊 **{symbol}**\n"
                        f"{signal_icon} **{bb_result['signal']}**\n"
                        f"💰 Preço: {bb_result['current_price']:.2f}"
                    )
        
        # Send alerts
        if alerts_to_send:
            message = "🚨 **ALERTAS OVECCHIA TRADING**\n\n" + "\n\n".join(alerts_to_send)
            try:
                await context.bot.send_message(chat_id=int(user_id), text=message, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Error sending alert to user {user_id}: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = """
🤖 **Comandos Disponíveis:**

/start - Menu principal
/screening - Configurar mudanças de estado  
/topobottom - Configurar topos e fundos
/estrategia - Escolher estratégia
/ativos - Configurar ativos
/status - Ver configurações atuais
/help - Esta mensagem
/stop - Parar alertas

📋 **Como usar:**
1. Use /start para ver o menu
2. Escolha o tipo de alerta
3. Selecione a estratégia
4. Envie os tickers (ex: BTC-USD, PETR4.SA)
5. Receba alertas automáticos!
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def stop_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop alerts for user"""
    users = load_users()
    user_id = str(update.effective_user.id)
    
    if user_id in users:
        users[user_id]['alerts_enabled'] = False
        save_users(users)
        await update.message.reply_text("⏹️ **Alertas desativados!** Use /start para reativar.")
    else:
        await update.message.reply_text("❌ Você não possui alertas configurados.")

def main():
    """Main function to run the bot"""
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN não encontrado nas variáveis de ambiente!")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stop", stop_alerts))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Schedule periodic alerts (every 15 minutes)
    job_queue = application.job_queue
    job_queue.run_repeating(send_alerts, interval=900, first=10)  # 900 seconds = 15 minutes
    
    print("🤖 Bot iniciado! Pressione Ctrl+C para parar.")
    
    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
