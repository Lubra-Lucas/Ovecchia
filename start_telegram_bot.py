
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
    # Set the bot token as environment variable
    os.environ['TELEGRAM_BOT_TOKEN'] = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    
    print("✅ Token do bot configurado automaticamente!")
    print("🤖 Bot: @Ovecchia_bot")
    print("🔗 Link direto: https://t.me/Ovecchia_bot")
    
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
