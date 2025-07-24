
import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-telegram-bot==20.7"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def start_bot():
    """Start the Telegram bot"""
    # Check if token is set
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        print("❌ Token do bot não configurado!")
        print("📋 Para configurar:")
        print("1. Crie um bot com @BotFather no Telegram")
        print("2. Configure a variável de ambiente TELEGRAM_BOT_TOKEN")
        print("3. Execute este script novamente")
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Start the bot
    try:
        import telegram_bot
        telegram_bot.main()
    except ImportError as e:
        print(f"❌ Erro ao importar o bot: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao iniciar o bot: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando OVECCHIA TRADING Telegram Bot...")
    start_bot()
