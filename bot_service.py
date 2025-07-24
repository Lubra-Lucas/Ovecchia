
import threading
import time
import os
import sys
import subprocess
import asyncio
from telegram_bot import main as run_bot

class BotService:
    def __init__(self):
        self.bot_thread = None
        self.is_running = False
        self.bot_process = None
        
    def start_bot_service(self):
        """Inicia o serviço do bot em uma thread separada"""
        if self.is_running:
            print("⚠️ Bot já está rodando!")
            return False
            
        # Configurar token do bot
        os.environ['TELEGRAM_BOT_TOKEN'] = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
        
        # Instalar dependências se necessário
        try:
            import telegram
            print("✅ python-telegram-bot já instalado")
        except ImportError:
            print("📦 Instalando python-telegram-bot...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "python-telegram-bot==20.7"])
            print("✅ Dependências instaladas!")
        
        # Iniciar bot em thread separada
        self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self.bot_thread.start()
        self.is_running = True
        print("🤖 Serviço do bot iniciado em background!")
        print("📱 Bot ativo e pronto para receber mensagens!")
        print("🔗 Acesse: https://t.me/Ovecchia_bot")
        return True
    
    def _run_bot(self):
        """Executa o bot em loop"""
        try:
            print("🚀 Iniciando polling do bot...")
            run_bot()
        except Exception as e:
            print(f"❌ Erro no bot: {e}")
            import traceback
            print(traceback.format_exc())
            self.is_running = False
    
    def stop_bot_service(self):
        """Para o serviço do bot"""
        self.is_running = False
        if self.bot_process:
            self.bot_process.terminate()
        print("⏹️ Serviço do bot parado!")
    
    def get_status(self):
        """Retorna o status do bot"""
        thread_alive = self.bot_thread.is_alive() if self.bot_thread else False
        return {
            'running': self.is_running,
            'thread_alive': thread_alive,
            'bot_username': '@Ovecchia_bot',
            'status_text': '🟢 Ativo e respondendo' if (self.is_running and thread_alive) else '🔴 Inativo'
        }

# Instância global do serviço
bot_service = BotService()

def start_telegram_service():
    """Função para iniciar o serviço do Telegram"""
    return bot_service.start_bot_service()

def stop_telegram_service():
    """Função para parar o serviço do Telegram"""
    return bot_service.stop_bot_service()

def get_telegram_status():
    """Função para obter status do serviço do Telegram"""
    return bot_service.get_status()
