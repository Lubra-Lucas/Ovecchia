
#!/usr/bin/env python3
"""
Teste rápido para verificar se o bot está respondendo
"""
import asyncio
import sys
from telegram import Bot

async def quick_test():
    """Teste rápido e direto"""
    bot_token = '8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k'
    
    print("🔄 Testando bot...")
    
    try:
        bot = Bot(token=bot_token)
        me = await bot.get_me()
        
        print(f"✅ Bot online: @{me.username}")
        print(f"🆔 ID: {me.id}")
        print(f"👤 Nome: {me.first_name}")
        
        # Verificar se pode receber updates
        print("🔄 Verificando capacidade de receber mensagens...")
        
        # Get recent updates (sem consumir)
        try:
            updates = await bot.get_updates(limit=1, timeout=1)
            print(f"✅ Sistema de updates funcionando (últimas mensagens: {len(updates)})")
        except Exception as e:
            print(f"⚠️ Aviso nos updates: {e}")
        
        print("\n✅ BOT ESTÁ FUNCIONANDO!")
        print("📱 Teste mandando /start em: https://t.me/Ovecchia_bot")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
