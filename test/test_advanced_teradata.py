"""
Test avanzado de conectividad Teradata con VPN
Diagnóstico detallado de conectividad y configuración
"""

import asyncio
import os
import sys
import socket

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings

async def advanced_teradata_diagnostics():
    """Diagnóstico avanzado de conectividad Teradata."""
    
    print("🔍 DIAGNÓSTICO AVANZADO DE CONECTIVIDAD TERADATA")
    print("=" * 52)
    
    # 1. Información de red local
    print(f"\n🌐 1. Información de red local:")
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"   💻 Hostname: {hostname}")
        print(f"   🔗 IP Local: {local_ip}")
    except Exception as e:
        print(f"   ❌ Error obteniendo info local: {e}")
    
    # 2. Resolución DNS del host Teradata
    print(f"\n🔍 2. Resolución DNS:")
    try:
        teradata_ip = socket.gethostbyname(settings.teradata_host)
        print(f"   ✅ {settings.teradata_host} → {teradata_ip}")
    except Exception as e:
        print(f"   ❌ Error resolviendo {settings.teradata_host}: {e}")
        print(f"   💡 Probando con IP directa...")
        teradata_ip = "161.131.180.193"  # IP conocida
    
    # 3. Test de conectividad en puertos específicos de Teradata
    print(f"\n🔌 3. Test de puertos Teradata:")
    ports_to_test = [1025, 443, 1433, 22]  # Puertos comunes de Teradata
    
    for port in ports_to_test:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # 5 segundos timeout
            result = sock.connect_ex((teradata_ip, port))
            sock.close()
            
            if result == 0:
                print(f"   ✅ Puerto {port}: ABIERTO")
            else:
                print(f"   ❌ Puerto {port}: CERRADO/FILTRADO")
        except Exception as e:
            print(f"   ❌ Puerto {port}: Error - {e}")
    
    # 4. Test con timeout extendido
    print(f"\n⏰ 4. Test con timeout extendido (30s):")
    try:
        import teradatasql
        
        print("   🔄 Intentando conexión con timeout de 30 segundos...")
        connection = teradatasql.connect(
            host=settings.teradata_host,
            user=settings.teradata_user,
            password=settings.teradata_password,
            database=settings.teradata_database,
            connect_timeout=30,  # 30 segundos
            request_timeout=30   # 30 segundos para requests
        )
        
        print("   ✅ ¡CONEXIÓN EXITOSA con timeout extendido!")
        
        # Test básico
        with connection.cursor() as cursor:
            cursor.execute("SELECT USER")
            result = cursor.fetchone()
            print(f"   👤 Usuario conectado: {result[0]}")
            
            cursor.execute("SELECT DATABASE")
            result = cursor.fetchone()
            print(f"   🗄️  Database: {result[0]}")
            
            # Test de tabla específica
            cursor.execute("SELECT TOP 1 TableName FROM DBC.TablesV WHERE DatabaseName = ?", [settings.teradata_database])
            result = cursor.fetchone()
            if result:
                print(f"   📊 Primera tabla en {settings.teradata_database}: {result[0]}")
            else:
                print(f"   ⚠️  No se encontraron tablas en {settings.teradata_database}")
        
        connection.close()
        print("   ✅ Conexión cerrada correctamente")
        return True
        
    except Exception as e:
        print(f"   ❌ Error con timeout extendido: {str(e)}")
        return False
    
    # 5. Test de configuraciones alternativas
    print(f"\n🔧 5. Test de configuraciones alternativas:")
    
    # Intentar con diferentes configuraciones
    configs_to_test = [
        {"host": settings.teradata_host, "logmech": "TD2"},
        {"host": settings.teradata_host, "logmech": "LDAP"},
        {"host": teradata_ip, "logmech": "TD2"},  # Con IP directa
    ]
    
    for i, config in enumerate(configs_to_test, 1):
        try:
            print(f"   🔄 Config {i}: {config}")
            
            connection = teradatasql.connect(
                host=config["host"],
                user=settings.teradata_user,
                password=settings.teradata_password,
                database=settings.teradata_database,
                logmech=config.get("logmech", "TD2"),
                connect_timeout=15
            )
            
            print(f"   ✅ Config {i}: ¡EXITOSA!")
            connection.close()
            return True
            
        except Exception as e:
            print(f"   ❌ Config {i}: {str(e)[:100]}...")
    
    return False


async def test_mcp_with_real_connection():
    """Test del MCP Server con conexión real si está disponible."""
    
    print(f"\n🚀 6. TEST MCP CON CONEXIÓN REAL:")
    print("-" * 35)
    
    # Intentar modificar temporalmente el MCP server para usar conexión real
    try:
        from src.agents.explain_generator import ExplainGeneratorAgent
        
        agent = ExplainGeneratorAgent()
        await agent.initialize()
        
        # Test con consulta simple que debería funcionar
        test_queries = [
            "SELECT USER",
            "SELECT DATABASE", 
            "SELECT CURRENT_TIMESTAMP",
            "SELECT COUNT(*) FROM DBC.TablesV WHERE DatabaseName = 'DBC'"
        ]
        
        for query in test_queries:
            print(f"\n   🔍 Probando: {query}")
            result = await agent.generate_explain(query)
            
            if result.success and result.explain_plan.strip():
                print(f"   ✅ EXPLAIN exitoso")
                print(f"   📊 Plan: {result.explain_plan[:100]}...")
                
                # Si obtenemos un plan real, significa que la conexión funciona
                if any(word in result.explain_plan.lower() for word in ['step', 'retrieve', 'scan', 'amp']):
                    print(f"   🎯 ¡PLAN REAL DE TERADATA DETECTADO!")
                    break
            else:
                print(f"   ❌ Error: {result.error_message}")
        
        await agent.cleanup()
        
    except Exception as e:
        print(f"   ❌ Error en test MCP: {str(e)}")


async def main():
    """Función principal del diagnóstico."""
    
    connection_success = await advanced_teradata_diagnostics()
    
    if connection_success:
        print(f"\n🎉 ¡CONEXIÓN A TERADATA ESTABLECIDA!")
        print("=" * 40)
        print("✅ Tu VPN está funcionando correctamente")
        print("✅ Las credenciales son válidas")
        print("✅ El servidor Teradata está accesible")
        
        # Continuar con test MCP
        await test_mcp_with_real_connection()
        
    else:
        print(f"\n⚠️  CONEXIÓN A TERADATA NO DISPONIBLE")
        print("=" * 42)
        print("💡 Posibles soluciones:")
        print("   1. Verificar que la VPN esté correctamente conectada")
        print("   2. Contactar al equipo de IT/DBA para verificar:")
        print("      - Estado del servidor Teradata")
        print("      - Configuración de firewall")
        print("      - Credenciales y permisos")
        print("   3. Verificar configuración de VPN corporativa")


if __name__ == "__main__":
    asyncio.run(main())
