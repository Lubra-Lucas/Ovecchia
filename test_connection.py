
#!/usr/bin/env python3
"""
Teste simples de conexão do bot
"""
import asyncio
import os

async def test_connection():
    """Teste básico de conexão"""
    print("🔄 Testando conexão...")
    
    try:
        # Set token
        os.environ['TELEGRAM_BOT_TOKEN'] = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
        
        # Import telegram
        from telegram import Bot
        
        bot = Bot(token=os.environ['TELEGRAM_BOT_TOKEN'])
        me = await bot.get_me()
        
        print(f"✅ Bot conectado!")
        print(f"👤 Nome: {me.first_name}")
        print(f"🤖 Username: @{me.username}")
        print(f"🆔 ID: {me.id}")
        print(f"🔗 Link: https://t.me/{me.username}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    print("✅ Teste concluído!" if success else "❌ Teste falhou!")
