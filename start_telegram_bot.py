
import os
import sys
import subprocess
import asyncio

def install_dependencies():
    """Install required dependencies"""
    try:
        print("📦 Verificando dependências...")
        # Try importing first
        try:
            import telegram
            print("✅ python-telegram-bot já disponível!")
            return True
        except ImportError:
            print("📦 Instalando python-telegram-bot...")
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
        print(f"✅ Bot conectado: @{me.username} (ID: {me.id})")
        return True
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False

def start_bot():
    """Start the Telegram bot"""
    print("🚀 Iniciando OVECCHIA TRADING Telegram Bot...")
    
    # Set the bot token as environment variable
    os.environ['TELEGRAM_BOT_TOKEN'] = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    
    print("✅ Token do bot configurado!")
    print("🤖 Bot: @Ovecchia_bot")
    print("🔗 Link: https://t.me/Ovecchia_bot")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Falha na instalação das dependências!")
        return False
    
    # Test connection
    print("🔄 Testando conexão...")
    try:
        connection_ok = asyncio.run(test_bot_connection())
        if not connection_ok:
            print("❌ Falha na conexão com o Telegram!")
            return False
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False
    
    # Start the bot
    try:
        print("\n🚀 Iniciando bot...")
        print("📱 Bot ativo e aguardando mensagens!")
        print("💬 Teste enviando /start no chat do bot")
        print("🔗 https://t.me/Ovecchia_bot")
        print("\n" + "="*50)
        
        import telegram_bot
        telegram_bot.main()
        
    except ImportError as e:
        print(f"❌ Erro ao importar telegram_bot: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Bot parado pelo usuário")
        return True
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    start_bot()
