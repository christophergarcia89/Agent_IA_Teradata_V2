"""
Test script para verificar la conexión con Azure OpenAI
"""

import asyncio
import os
import sys

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.azure_openai_utils import azure_openai_manager

async def test_azure_connection():
    """Test de conexión con Azure OpenAI."""
    print("🧪 Probando conexión con Azure OpenAI...")
    
    try:
        # Test de conexión básica
        is_connected = azure_openai_manager.test_connection()
        
        if is_connected:
            print("✅ Conexión exitosa con Azure OpenAI")
            
            # Test de generación de respuesta
            print("\n🧪 Probando generación de respuesta...")
            llm = azure_openai_manager.get_llm()
            
            test_messages = [
                ("system", "Eres un experto en SQL para Teradata."),
                ("human", "¿Cuál es la diferencia entre INNER JOIN y LEFT JOIN?")
            ]
            
            response = await llm.ainvoke(test_messages)
            print(f"✅ Respuesta generada exitosamente")
            print(f"📝 Respuesta: {response.content[:200]}...")
            
        else:
            print("❌ Error en la conexión con Azure OpenAI")
            
    except Exception as e:
        print(f"❌ Error durante las pruebas: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_azure_connection())
