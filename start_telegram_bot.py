
import os
import sys
import subprocess
import asyncio

def install_dependencies():
    """Install required dependencies"""
    try:
        print("📦 Instalando dependências...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-telegram-bot==20.7"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

async def test_bot_connection():
    """Test bot connection before starting"""
    try:
        from telegram import Bot
        bot_token = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
        bot = Bot(token=bot_token)
        
        me = await bot.get_me()
        print(f"✅ Bot conectado: @{me.username}")
        return True
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False

def start_bot():
    """Start the Telegram bot"""
    # Set the bot token as environment variable
    os.environ['TELEGRAM_BOT_TOKEN'] = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    
    print("✅ Token do bot configurado automaticamente!")
    print("🤖 Bot: @Ovecchia_bot")
    print("🔗 Link direto: https://t.me/Ovecchia_bot")
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test connection
    print("🔄 Testando conexão com Telegram...")
    try:
        connection_ok = asyncio.run(test_bot_connection())
        if not connection_ok:
            return False
    except Exception as e:
        print(f"❌ Erro no teste de conexão: {e}")
        return False
    
    # Start the bot
    try:
        print("🚀 Iniciando bot e mantendo ativo...")
        print("📱 Bot pronto para receber mensagens!")
        print("💬 Envie /start no Telegram para testar")
        
        import telegram_bot
        # This will keep the bot running indefinitely
        telegram_bot.main()
        
    except ImportError as e:
        print(f"❌ Erro ao importar o bot: {e}")
        return False
    except SyntaxError as e:
        print(f"❌ Erro de sintaxe no bot: {e}")
        print("🔧 Verifique o arquivo telegram_bot.py")
        return False
    except KeyboardInterrupt:
        print("⏹️ Bot parado pelo usuário")
        return True
    except Exception as e:
        print(f"❌ Erro ao executar o bot: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🚀 Iniciando OVECCHIA TRADING Telegram Bot...")
    start_bot()
