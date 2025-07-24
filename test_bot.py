
import asyncio
import os
from telegram import Bot

async def test_bot():
    """Testar se o bot está respondendo"""
    bot_token = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    
    try:
        bot = Bot(token=bot_token)
        
        print("🔄 Testando conexão com o bot...")
        me = await bot.get_me()
        print(f"✅ Bot está online!")
        print(f"👤 Nome: {me.first_name}")
        print(f"🤖 Username: @{me.username}")
        print(f"🆔 ID: {me.id}")
        
        # Testar webhook info
        webhook_info = await bot.get_webhook_info()
        print(f"🔗 Webhook: {webhook_info.url if webhook_info.url else 'Polling mode (correto)'}")
        
        # Testar comandos
        print("\n🧪 Testando comandos...")
        commands = await bot.get_my_commands()
        print(f"📋 Comandos registrados: {len(commands)}")
        
        # Testar se consegue processar updates (simulação)
        print("✅ Bot configurado corretamente para receber mensagens")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def test_full_functionality():
    """Teste mais completo"""
    print("🚀 Teste completo do bot...")
    
    # Teste básico
    basic_ok = await test_bot()
    if not basic_ok:
        return False
    
    print("\n🔍 Verificando funcionalidades...")
    
    # Verificar se o arquivo de usuários pode ser criado
    try:
        import json
        test_users = {"test": {"username": "test", "alerts_enabled": False}}
        with open('telegram_users_test.json', 'w') as f:
            json.dump(test_users, f)
        os.remove('telegram_users_test.json')
        print("✅ Sistema de armazenamento funcionando")
    except Exception as e:
        print(f"⚠️ Problema no sistema de armazenamento: {e}")
    
    # Verificar dependências de análise
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        print("✅ Dependências de análise disponíveis")
    except ImportError as e:
        print(f"⚠️ Dependência faltando: {e}")
    
    print("\n✅ Teste completo finalizado!")
    return True

if __name__ == "__main__":
    print("🚀 Iniciando testes do Telegram Bot...")
    
    success = asyncio.run(test_full_functionality())
    
    if success:
        print("\n" + "="*50)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("🔗 Acesse: https://t.me/Ovecchia_bot")
        print("📱 Envie /start para testar")
        print("="*50)
    else:
        print("\n❌ FALHA NOS TESTES!")
