
import threading
import time
import os
import sys
import subprocess
import asyncio

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
        
        # Iniciar bot usando subprocess para evitar imports circulares
        try:
            print("🚀 Iniciando bot como processo separado...")
            self.bot_process = subprocess.Popen(
                [sys.executable, "start_telegram_bot.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Aguardar um pouco para verificar se iniciou corretamente
            time.sleep(3)
            
            if self.bot_process.poll() is None:  # Processo ainda rodando
                self.is_running = True
                print("🤖 Serviço do bot iniciado com sucesso!")
                print("📱 Bot ativo e pronto para receber mensagens!")
                print("🔗 Acesse: https://t.me/Ovecchia_bot")
                return True
            else:
                # Processo falhou
                stdout, stderr = self.bot_process.communicate()
                print(f"❌ Falha ao iniciar bot: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao iniciar bot: {e}")
            return False
    
    def stop_bot_service(self):
        """Para o serviço do bot"""
        self.is_running = False
        if self.bot_process and self.bot_process.poll() is None:
            self.bot_process.terminate()
            try:
                self.bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.bot_process.kill()
        print("⏹️ Serviço do bot parado!")
    
    def get_status(self):
        """Retorna o status do bot"""
        process_alive = self.bot_process and self.bot_process.poll() is None
        return {
            'running': self.is_running and process_alive,
            'process_alive': process_alive,
            'bot_username': '@Ovecchia_bot',
            'status_text': '🟢 Ativo e respondendo' if (self.is_running and process_alive) else '🔴 Inativo'
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
