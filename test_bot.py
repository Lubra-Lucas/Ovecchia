
#!/usr/bin/env python3
"""
Script de teste para verificar o funcionamento do bot do Telegram
"""
import requests
import os
import sys

def test_telegram_bot():
    """Testa a conectividade e funcionamento básico do bot"""
    
    # Token do bot
    BOT_TOKEN = "8487471783:AAElQBvIhVcbtVmEoPEdnuafMUR4mwGJh1k"
    
    print("🧪 TESTE DE FUNCIONAMENTO DO BOT")
    print("=" * 50)
    
    # Teste 1: Verificar se o token é válido
    print("\n1. 🔑 Testando token do bot...")
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get('ok'):
                bot_data = bot_info['result']
                print(f"   ✅ Token válido!")
                print(f"   🤖 Nome: {bot_data.get('first_name')}")
                print(f"   📧 Username: @{bot_data.get('username')}")
                print(f"   🆔 ID: {bot_data.get('id')}")
            else:
                print(f"   ❌ Resposta inválida da API: {bot_info}")
                return False
        else:
            print(f"   ❌ Erro HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro na conexão: {str(e)}")
        return False
    
    # Teste 2: Verificar updates recentes
    print("\n2. 📨 Verificando mensagens recentes...")
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            updates = response.json()
            if updates.get('ok'):
                recent_updates = updates.get('result', [])
                print(f"   ✅ API funcionando - {len(recent_updates)} updates encontrados")
                
                if recent_updates:
                    latest_update = recent_updates[-1]
                    update_id = latest_update.get('update_id')
                    print(f"   📝 Último update ID: {update_id}")
                    
                    if 'message' in latest_update:
                        msg = latest_update['message']
                        from_user = msg.get('from', {})
                        text = msg.get('text', 'N/A')
                        print(f"   💬 Última mensagem: '{text}' de {from_user.get('first_name', 'Unknown')}")
                else:
                    print("   📭 Nenhuma mensagem recente encontrada")
            else:
                print(f"   ❌ Erro na resposta: {updates}")
                return False
        else:
            print(f"   ❌ Erro HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro ao verificar updates: {str(e)}")
        return False
    
    # Teste 3: Verificar webhook (se configurado)
    print("\n3. 🔗 Verificando configuração de webhook...")
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getWebhookInfo"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            webhook_info = response.json()
            if webhook_info.get('ok'):
                webhook_data = webhook_info['result']
                webhook_url = webhook_data.get('url', '')
                
                if webhook_url:
                    print(f"   ⚠️ Webhook configurado: {webhook_url}")
                    print("   ⚠️ ATENÇÃO: Bot pode estar em modo webhook, não polling!")
                    print("   💡 Para usar polling, desative o webhook primeiro")
                    
                    # Tentar desativar webhook
                    print("\n   🔧 Tentando desativar webhook...")
                    delete_url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
                    delete_response = requests.post(delete_url, timeout=10)
                    
                    if delete_response.status_code == 200:
                        delete_result = delete_response.json()
                        if delete_result.get('ok'):
                            print("   ✅ Webhook desativado com sucesso!")
                        else:
                            print(f"   ❌ Erro ao desativar webhook: {delete_result}")
                    else:
                        print(f"   ❌ Erro HTTP ao desativar webhook: {delete_response.status_code}")
                else:
                    print("   ✅ Nenhum webhook configurado - modo polling OK")
            else:
                print(f"   ❌ Erro na resposta: {webhook_info}")
                return False
        else:
            print(f"   ❌ Erro HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro ao verificar webhook: {str(e)}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ TESTE CONCLUÍDO - BOT PARECE ESTAR FUNCIONANDO!")
    print("\n💡 PRÓXIMOS PASSOS:")
    print("1. Execute: python telegram_bot.py")
    print("2. Envie /start no Telegram para @Ovecchia_bot")
    print("3. Verifique os logs no console")
    
    return True

if __name__ == "__main__":
    test_telegram_bot()
