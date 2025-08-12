"""
Servidor MCP para conectividad con Teradata
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Importar teradatasql para conexión real
try:
    import teradatasql
    TERADATASQL_AVAILABLE = True
    print("✅ teradatasql library available")
except ImportError:
    TERADATASQL_AVAILABLE = False
    print("❌ teradatasql library not available - using simulation mode")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Teradata MCP Server (Real Connection)",
    description="Model Context Protocol Server for Teradata SQL execution with real teradatasql connection",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    """Modelo para solicitudes de ejecución de consulta."""
    query: str
    database_uri: str

class QueryResponse(BaseModel):
    """Modelo para respuestas de consulta."""
    success: bool
    query: str
    rows: Optional[list] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: str

def parse_database_uri(uri: str) -> dict:
    """
    Parse DATABASE_URI format: teradata://user:password@host/database
    """
    try:
        # Remover prefijo teradata:// (11 caracteres)
        if uri.startswith('teradata://'):
            uri = uri[11:]
        
        # Separar credenciales y host/database
        if '@' in uri:
            credentials, host_db = uri.split('@', 1)
            if ':' in credentials:
                user, password = credentials.split(':', 1)
            else:
                user = credentials
                password = ""
        else:
            raise ValueError("Invalid URI format")
        
        # Separar host y database (y manejar parámetros de query)
        if '/' in host_db:
            host, database_params = host_db.split('/', 1)
            # Separar database de parámetros de query si existen
            if '?' in database_params:
                database, query_params = database_params.split('?', 1)
            else:
                database = database_params
        else:
            # Solo host (puede tener parámetros)
            if '?' in host_db:
                host, query_params = host_db.split('?', 1)
            else:
                host = host_db
            database = "DBC"
        
        # Limpiar puerto si está incluido en host para mostrar solo host
        if ':' in host:
            host_only = host.split(':')[0]
        else:
            host_only = host
        
        return {
            "host": host,  # Host completo con puerto
            "host_only": host_only,  # Solo hostname/IP sin puerto
            "user": user,
            "password": password,
            "database": database
        }
    except Exception as e:
        raise ValueError(f"Error parsing DATABASE_URI: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud."""
    return {
        "status": "healthy",
        "service": "Teradata MCP Server (Real Connection)" if TERADATASQL_AVAILABLE else "Teradata MCP Server (Simulation)",
        "version": "2.0.0",
        "teradatasql_available": TERADATASQL_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "Teradata MCP Server with Real Connection",
        "teradatasql_available": TERADATASQL_AVAILABLE,
        "endpoints": [
            "/health - Health check",
            "/execute - Execute SQL query",
            "/docs - API documentation"
        ]
    }

@app.post("/execute", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Ejecutar una consulta SQL en Teradata.
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Executing query: {request.query[:100]}...")
        
        if TERADATASQL_AVAILABLE:
            # Usar conexión real con teradatasql
            result = await execute_real_query(request.query, request.database_uri)
        else:
            # Fallback a simulación
            result = await simulate_query_execution(request.query, request.database_uri)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            success=True,
            query=request.query,
            rows=result.get("rows", []),
            metadata={
                "execution_time": execution_time,
                "row_count": len(result.get("rows", [])),
                "connection_type": "real" if TERADATASQL_AVAILABLE else "simulated"
            },
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            success=False,
            query=request.query,
            error=error_msg,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )

async def execute_real_query(query: str, database_uri: str) -> dict:
    """
    Ejecutar consulta real usando teradatasql.
    """
    connection = None
    try:
        # Parse URI
        conn_params = parse_database_uri(database_uri)
        logger.info(f"Connecting to {conn_params['host_only']} (port implicit) as {conn_params['user']}")
        
        # Crear conexión usando solo hostname/IP sin puerto (el puerto se maneja automáticamente)
        connection = teradatasql.connect(
            host=conn_params['host_only'],  # Solo IP/hostname sin puerto
            user=conn_params['user'],
            password=conn_params['password'],
            connect_timeout=10000  # 10 segundos en milisegundos
        )
        
        # Ejecutar consulta
        with connection.cursor() as cursor:
            cursor.execute(query)
            
            # Obtener resultados
            if cursor.description:
                # Consulta que retorna datos
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                # Convertir a formato lista de diccionarios
                result_rows = []
                for row in rows:
                    result_rows.append(dict(zip(columns, row)))
                
                return {"rows": result_rows}
            else:
                # Consulta que no retorna datos (INSERT, UPDATE, etc.)
                return {"rows": [{"status": "Query executed successfully"}]}
                
    except Exception as e:
        logger.error(f"Real query execution failed: {str(e)}")
        raise Exception(f"Teradata connection error: {str(e)}")
    
    finally:
        if connection:
            try:
                connection.close()
            except:
                pass

async def simulate_query_execution(query: str, database_uri: str) -> dict:
    """
    Simular ejecución de consulta (fallback cuando teradatasql no está disponible).
    """
    logger.info(f"Simulating query execution for: {query[:50]}...")
    
    # Simular delay
    await asyncio.sleep(0.2)
    
    # Parse URI para mostrar información
    try:
        conn_params = parse_database_uri(database_uri)
        host_info = f"{conn_params['host']}/{conn_params['database']}"
    except:
        host_info = "simulated_host"
    
    # Generar respuesta simulada basada en el tipo de consulta
    query_upper = query.upper().strip()
    
    if query_upper.startswith('EXPLAIN'):
        # Simular plan EXPLAIN
        return {
            "rows": [
                {"explain_plan": f"Simulated EXPLAIN plan for query on {host_info}"},
                {"step": "1) First, we lock database for reading"},
                {"step": "2) We scan the specified tables"},
                {"step": "3) We apply WHERE conditions"},
                {"step": "4) We return the result set"},
                {"metadata": f"Estimated cost: 0.5 seconds, Connection: {host_info}"}
            ]
        }
    elif query_upper.startswith('SELECT USER'):
        return {"rows": [{"User": conn_params.get('user', 'simulated_user')}]}
    elif query_upper.startswith('SELECT DATABASE'):
        return {"rows": [{"Database": conn_params.get('database', 'simulated_db')}]}
    elif query_upper.startswith('SELECT COUNT'):
        return {"rows": [{"count": 1000}]}
    elif query_upper.startswith('SELECT'):
        # Simular SELECT genérico
        return {
            "rows": [
                {"column1": "value1", "column2": "value2"},
                {"column1": "value3", "column2": "value4"}
            ]
        }
    else:
        # Otras consultas
        return {"rows": [{"status": "Query simulated successfully", "connection": host_info}]}

@app.post("/test-connection")
async def test_connection_endpoint(request: QueryRequest):
    """
    Probar la conexión a Teradata.
    """
    try:
        logger.info("Testing Teradata connection...")
        
        if TERADATASQL_AVAILABLE:
            # Test real
            conn_params = parse_database_uri(request.database_uri)
            
            connection = teradatasql.connect(
                host=conn_params['host'],
                user=conn_params['user'],
                password=conn_params['password'],
                database=conn_params['database'],
                connect_timeout=10
            )
            
            # Test básico
            with connection.cursor() as cursor:
                cursor.execute("SELECT USER")
                result = cursor.fetchone()
                current_user = result[0] if result else "unknown"
            
            connection.close()
            
            return {
                "success": True,
                "message": "Real connection test successful",
                "connection_type": "real",
                "current_user": current_user,
                "host": conn_params['host'],
                "database": conn_params['database'],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Test simulado
            await asyncio.sleep(0.1)
            conn_params = parse_database_uri(request.database_uri)
            
            return {
                "success": True,
                "message": "Simulated connection test",
                "connection_type": "simulated",
                "host": conn_params['host'],
                "database": conn_params['database'],
                "user": conn_params['user'],
                "note": "teradatasql library not available - using simulation",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Connection test failed: {str(e)}"
        )

def run_server():
    """Ejecutar el servidor MCP."""
    print("🚀 Iniciando Teradata MCP Server (Real Connection)...")
    print("📍 URL: http://localhost:3002")
    print("📚 Documentación: http://localhost:3002/docs")
    print("💚 Health Check: http://localhost:3002/health")
    
    if TERADATASQL_AVAILABLE:
        print("\n✅ teradatasql library available - REAL connections enabled")
        print("🔗 This server will attempt real connections to Teradata")
    else:
        print("\n⚠️  teradatasql library not available - using SIMULATION mode")
        print("💡 Install teradatasql for real connections: pip install teradatasql")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3002,  # Puerto diferente para no conflictuar
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
