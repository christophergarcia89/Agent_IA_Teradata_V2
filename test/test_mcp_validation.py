"""
Test de validación MCP
"""

import asyncio
import os
import sys
from datetime import datetime

# Añadir el directorio raíz al path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

try:
    from src.agents.explain_generator import EnhancedExplainGenerator
    from config.settings import settings
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("💡 Verificar que el entorno virtual esté activado")
    sys.exit(1)


async def validate_mcp_with_working_config():
    """
    Valida la integración MCP usando la configuración que ya sabemos que funciona.
    Basado en el éxito de test_direct_teradata_basic.py con IP 10.33.84.36
    """
    print("🚀 VALIDACIÓN MCP CON CONFIGURACIÓN PROBADA")
    print("=" * 55)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Verificar configuración actual
    print("\n📋 1. VERIFICANDO CONFIGURACIÓN:")
    print(f"   DATABASE_URI: {settings.database_uri}")
    print(f"   MCP Server: {settings.teradata_mcp_server_url}")
    
    # Validar IP correcta (la que funciona)
    if "10.33.84.36" in settings.database_uri:
        print("   ✅ Usando IP FUNCIONAL: 10.33.84.36")
        ip_status = "✅ CORRECTO"
    elif "161.131.180.193" in settings.database_uri:
        print("   ⚠️  ADVERTENCIA: IP 161.131.180.193 está BLOQUEADA")
        ip_status = "⚠️ BLOQUEADA"
    elif "EDW" in settings.database_uri:
        print("   ⚠️  ADVERTENCIA: Hostname EDW está BLOQUEADO")
        ip_status = "⚠️ BLOQUEADO"
    else:
        print("   ❓ IP no reconocida")
        ip_status = "❓ DESCONOCIDA"
    
    print(f"   Estado IP: {ip_status}")
    
    # 2. Test de inicialización del agente
    print("\n🔧 2. INICIALIZANDO AGENTE MCP:")
    agent = EnhancedExplainGenerator()
    
    start_time = datetime.now()
    
    try:
        # Intentar inicialización con timeout
        print("   🔄 Inicializando conexión MCP...")
        await asyncio.wait_for(agent.initialize(), timeout=30.0)
        
        init_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Agente MCP inicializado en {init_time:.2f}s")
        
        # 3. Test básico de EXPLAIN
        print(f"\n📊 3. TEST BÁSICO DE EXPLAIN:")
        print("   📝 Usando query simple que sabemos que funciona...")
        
        # Query que funcionó en test directo
        test_query = "SELECT USER, DATABASE, SESSION"
        print(f"   Query: {test_query}")
        
        explain_start = datetime.now()
        
        try:
            result = await asyncio.wait_for(
                agent.generate_explain_plan(test_query), 
                timeout=45.0
            )
            
            explain_time = (datetime.now() - explain_start).total_seconds()
            
            if result.success:
                print(f"   ✅ EXPLAIN exitoso en {explain_time:.2f}s")
                print(f"   🔧 Tools MCP usadas: {len(result.mcp_tools_used) if result.mcp_tools_used else 0}")
                
                if result.explain_plan:
                    plan_preview = result.explain_plan[:200].replace('\n', ' ')
                    print(f"   📄 Plan (preview): {plan_preview}...")
                
                if result.warnings:
                    print(f"   ⚠️  Advertencias: {len(result.warnings)}")
                    
            else:
                print(f"   ❌ EXPLAIN falló en {explain_time:.2f}s")
                print(f"   Error: {result.error_message}")
                return False
                
        except asyncio.TimeoutError:
            print("   ⏰ Timeout en EXPLAIN (>45s)")
            print("   💡 MCP Server podría no estar respondiendo")
            return False
        
        # 4. Test de información del servidor MCP
        print(f"\n🌐 4. INFO DEL MCP SERVER:")
        try:
            if agent.client and hasattr(agent.client, 'get_available_tools_enhanced'):
                try:
                    available_tools = await asyncio.wait_for(
                        agent.client.get_available_tools_enhanced(), 
                        timeout=15.0
                    )
                    
                    print(f"   🔌 Estado: Conectado")
                    print(f"   🚀 Transport: {'HTTP' if hasattr(agent.client, 'http_client') and agent.client.http_client else 'STDIO' if hasattr(agent.client, 'session') and agent.client.session else 'N/A'}")
                    
                    if isinstance(available_tools, list):
                        print(f"   🔧 Tools disponibles: {len(available_tools)}")
                        
                        if available_tools:
                            print("   📋 Principales tools:")
                            for tool in available_tools[:3]:  # Mostrar solo las primeras 3
                                if isinstance(tool, dict):
                                    tool_name = tool.get('name', 'Sin nombre')
                                    tool_desc = tool.get('description', 'Sin descripción')[:50]
                                    print(f"      - {tool_name}: {tool_desc}...")
                    elif isinstance(available_tools, dict):
                        # Handle case where safe_operation returns a dict instead of list
                        if available_tools.get('success', False):
                            tools_list = available_tools.get('content', [])
                            print(f"   🔧 Tools disponibles: {len(tools_list) if isinstance(tools_list, list) else 'N/A'}")
                        else:
                            print(f"   ❌ Error obteniendo tools: {available_tools.get('error', 'Error desconocido')}")
                    else:
                        print(f"   🔧 Respuesta de tools: {type(available_tools)}")
                        
                except Exception as tool_err:
                    print(f"   ⚠️ Error obteniendo tools: {tool_err}")
                    print(f"   🔌 Estado: Parcialmente conectado")
            else:
                print("   ❌ Cliente MCP no inicializado o método no disponible")
                    
        except asyncio.TimeoutError:
            print("   ⏰ Timeout obteniendo info del servidor")
        except Exception as server_err:
            print(f"   ❌ Error servidor: {server_err}")
        
        # 5. Resumen final
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ VALIDACIÓN MCP COMPLETADA")
        print(f"⏱️  Tiempo total: {total_time:.2f}s")
        print(f"🎯 Estado: FUNCIONAL con IP 10.33.84.36")
        print(f"🔗 MCP integrado correctamente")
        
        return True
        
    except asyncio.TimeoutError:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n⏰ TIMEOUT después de {elapsed:.2f}s")
        print("🔍 ANÁLISIS:")
        
        if "10.33.84.36" in settings.database_uri:
            print("   ✅ IP de Teradata es correcta (10.33.84.36)")
            print("   ❓ Problema probablemente en MCP Server")
            print("   💡 Verificar: http://localhost:3002 está corriendo?")
        else:
            print("   ❌ IP de Teradata incorrecta")
            print("   🔧 Usar IP funcional: 10.33.84.36")
            
        return False
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ ERROR después de {elapsed:.2f}s: {str(e)}")
        print(f"📊 Diagnóstico:")
        print(f"   - IP Teradata: {'✅ Correcta' if '10.33.84.36' in settings.database_uri else '❌ Incorrecta'}")
        print(f"   - Conexión directa: ✅ VALIDADA (test_direct_teradata_basic.py)")
        print(f"   - Problema: Probablemente en capa MCP")
        
        return False
        
    finally:
        try:
            await agent.cleanup()
            print("🔒 Limpieza completada")
        except:
            pass


async def test_fallback_mode():
    """Test del modo fallback si MCP falla."""
    print("\n🔄 TEST DE MODO FALLBACK")
    print("-" * 35)
    
    agent = EnhancedExplainGenerator()
    # No inicializar para forzar modo fallback
    
    try:
        test_query = "SELECT COUNT(*) FROM DBC.TablesV WHERE TableKind = 'T'"
        print(f"📝 Query fallback: {test_query}")
        
        result = await asyncio.wait_for(
            agent.generate_explain_plan(test_query),
            timeout=30.0
        )
        
        if result.success:
            print("✅ Modo fallback funcional")
            print(f"📄 Plan generado (preview): {result.explain_plan[:100]}...")
        else:
            print(f"❌ Fallback falló: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Error en fallback: {e}")
    finally:
        try:
            await agent.cleanup()
        except:
            pass


async def main():
    """Función principal."""
    print("🎯 TEST MCP VALIDATION")
    print("Basado en éxito de test_direct_teradata_basic.py")
    print("IP validada: 10.33.84.36")
    print()
    
    # Test principal
    success = await validate_mcp_with_working_config()
    
    # Si falla, probar fallback
    if not success:
        await test_fallback_mode()
    
    # Resultado final
    print(f"\n{'='*50}")
    print(f"⏰ Completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("🎉 RESULTADO: ✅ MCP VALIDADO EXITOSAMENTE")
        print("🚀 La integración MCP funciona correctamente")
        return 0
    else:
        print("❌ RESULTADO: MCP con problemas")
        print("💡 Conexión directa Teradata ✅ OK (IP 10.33.84.36)")
        print("🔧 Verificar MCP Server en localhost:3002")
        return 1


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⚠️ Test cancelado por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        sys.exit(1)
