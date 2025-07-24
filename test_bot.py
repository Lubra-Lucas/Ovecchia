
import asyncio
import os
from telegram import Bot

async def test_bot():
    """Testar se o bot está respondendo"""
    bot_token = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    
    try:
        bot = Bot(token=bot_token)
        
        # Testar se o bot está online
        me = await bot.get_me()
        print(f"✅ Bot está online!")
        print(f"👤 Nome: {me.first_name}")
        print(f"🤖 Username: @{me.username}")
        print(f"🆔 ID: {me.id}")
        
        # Testar webhook info (se houver)
        webhook_info = await bot.get_webhook_info()
        print(f"🔗 Webhook URL: {webhook_info.url if webhook_info.url else 'Nenhum webhook configurado'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao conectar com o bot: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_bot())
