
import asyncio
import os
from telegram import Bot

async def test_bot():
    """Testar se o bot está respondendo"""
    bot_token = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    
    try:
        bot = Bot(token=bot_token)
        
        # Testar se o bot está online
        print("🔄 Testando conexão com o bot...")
        me = await bot.get_me()
        print(f"✅ Bot está online!")
        print(f"👤 Nome: {me.first_name}")
        print(f"🤖 Username: @{me.username}")
        print(f"🆔 ID: {me.id}")
        
        # Testar webhook info (se houver)
        webhook_info = await bot.get_webhook_info()
        print(f"🔗 Webhook URL: {webhook_info.url if webhook_info.url else 'Nenhum webhook configurado'}")
        
        # Testar comandos básicos
        print("\n🧪 Testando comandos do bot...")
        commands = await bot.get_my_commands()
        print(f"📋 Comandos disponíveis: {len(commands)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao conectar com o bot: {e}")
        return False

async def send_test_message(chat_id=None):
    """Enviar mensagem de teste (opcional)"""
    if not chat_id:
        print("ℹ️ Para testar envio de mensagens, forneça um chat_id")
        return
        
    bot_token = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    bot = Bot(token=bot_token)
    
    try:
        await bot.send_message(
            chat_id=chat_id, 
            text="🤖 **Teste do OVECCHIA TRADING Bot**\n\nBot funcionando corretamente!",
            parse_mode='Markdown'
        )
        print(f"✅ Mensagem de teste enviada para {chat_id}")
    except Exception as e:
        print(f"❌ Erro ao enviar mensagem: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando testes do Telegram Bot...")
    
    # Teste básico de conexão
    success = asyncio.run(test_bot())
    
    if success:
        print("\n✅ Todos os testes passaram!")
        print("🔗 Acesse o bot em: https://t.me/Ovecchia_bot")
        print("📱 Envie /start para testar a funcionalidade completa")
    else:
        print("\n❌ Falha nos testes!")
