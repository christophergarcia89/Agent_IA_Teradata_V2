"""
Script para instalar dependencias del Teradata MCP Server.
"""

import subprocess
import sys
import logging

def install_mcp_dependencies():
    """Instala las dependencias necesarias para MCP."""
    
    dependencies = [
        "langchain-mcp-adapters",
        "mcp",
        "httpx",
        "uvloop",
    ]
    
    print("🔧 Instalando dependencias MCP para Teradata...")
    
    for dep in dependencies:
        try:
            print(f"📦 Instalando {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} instalado exitosamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {dep}: {e}")
            return False
    
    print("\n🚀 Instalando Teradata MCP Server...")
    try:
        # Instalar uv si no está disponible
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
        print("✅ uv instalado")
        
        # Instalar teradata-mcp-server
        subprocess.check_call(["uv", "add", "teradata-mcp-server"])
        print("✅ teradata-mcp-server instalado")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error con uv/teradata-mcp-server: {e}")
        print("💡 Puedes instalar manualmente con:")
        print("   pip install uv")
        print("   uv add teradata-mcp-server")
    
    print("\n📋 INSTRUCCIONES POST-INSTALACIÓN:")
    print("1. Configurar DATABASE_URI en .env:")
    print("   DATABASE_URI=teradata://user:password@host/database")
    print("\n2. Ejecutar MCP Server:")
    print("   export DATABASE_URI='teradata://user:password@host/database'")
    print("   uv run teradata-mcp-server")
    print("\n3. El servidor estará disponible en http://localhost:3000")
    
    return True

if __name__ == "__main__":
    install_mcp_dependencies()