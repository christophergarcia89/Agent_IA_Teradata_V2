"""
FastAPI web interface for the Teradata SQL Agent.
"""

import os
import sys
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Configure encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)
    # Force UTF-8 stdout
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.workflow import TeradataWorkflow, WorkflowResult
from src.utils.logging_utils import setup_logging, PerformanceLogger, SQLQueryLogger
from config.settings import settings


# Setup logging
setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Teradata SQL Agent",
    description="LangGraph-based SQL query review and optimization system",
    version="1.0.0"
)

# Initialize loggers
perf_logger = PerformanceLogger()
sql_logger = SQLQueryLogger()

# Initialize workflow (will be done on startup)
workflow: TeradataWorkflow = None

# Request/Response models
class QueryRequest(BaseModel):
    """Request model for SQL query analysis."""
    query: str
    query_id: str = None


class QueryResponse(BaseModel):
    """Response model for SQL query analysis."""
    query_id: str
    success: bool
    original_query: str
    corrected_query: str
    is_standards_compliant: bool
    standards_violations: List[str]
    explain_plan: str
    performance_assessment: str
    optimization_suggestions: List[Dict[str, Any]]
    final_recommendations: List[str]
    error_messages: List[str]
    processing_time: float


@app.on_event("startup")
async def startup_event():
    """Initialize minimal components only - Lazy workflow initialization."""
    global workflow
    try:
        # NO inicializar el workflow completo en startup
        # Solo crear la instancia sin inicializar
        workflow = TeradataWorkflow()
        
        # Marcar que no está inicializado
        workflow._initialized = False
        workflow._initializing = False
        
        # Log que estamos en modo lazy
        logging.info("Workflow created in LAZY mode - will initialize on first request")
        logging.info("This avoids VectorStore/ChromaDB blocking during startup")
        
    except Exception as e:
        logging.error(f"Failed to create workflow instance: {e}")
        # No hacer raise para que la app funcione aunque sea parcialmente


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    if workflow:
        await workflow.cleanup()
        logging.info("Workflow cleanup completed")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Teradata SQL Agent</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            textarea { width: 100%; height: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .results { margin-top: 30px; }
            .result-section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }
            .error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
            .success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
            .warning { background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }
            .loading { text-align: center; padding: 20px; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
            .suggestion { margin-bottom: 10px; padding: 10px; border-left: 4px solid #007bff; background-color: #f8f9fa; }
            .priority-critical { border-left-color: #dc3545; }
            .priority-high { border-left-color: #fd7e14; }
            .priority-medium { border-left-color: #ffc107; }
            .priority-low { border-left-color: #28a745; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>[SQL] Teradata SQL Agent</h1>
                <p>Análisis inteligente de queries SQL con RAG y optimización automática</p>
                <div id="systemStatus" class="warning" style="display: none;">
                    <p><strong>[ESTADO] Estado del sistema:</strong> <span id="statusText">Verificando...</span></p>
                </div>
            </div>
            
            <form id="queryForm">
                <div class="form-group">
                    <label for="sqlQuery">Query SQL a analizar:</label>
                    <textarea id="sqlQuery" name="query" placeholder="Ingresa tu query SQL aquí..." required></textarea>
                </div>
                <button type="submit">[ANALIZAR] Analizar Query</button>
            </form>
            
            <div id="results" class="results" style="display: none;"></div>
        </div>

        <script>
            // Check system status on page load
            async function checkSystemStatus() {
                try {
                    const response = await fetch('/health');
                    const health = await response.json();
                    const statusDiv = document.getElementById('systemStatus');
                    const statusText = document.getElementById('statusText');
                    
                    if (health.status === 'ready' && health.mode === 'lazy') {
                        statusDiv.style.display = 'block';
                        statusDiv.className = 'warning';
                        statusText.textContent = 'Sistema listo - Se inicializará completamente en la primera consulta (puede tomar 1-2 min)';
                    } else if (health.status === 'initializing') {
                        statusDiv.style.display = 'block';
                        statusDiv.className = 'loading';
                        statusText.textContent = 'Inicializando sistema completo... Esto puede tomar 1-2 minutos';
                    } else if (health.status === 'healthy') {
                        statusDiv.style.display = 'block';
                        statusDiv.className = 'success';
                        statusText.textContent = 'Sistema completamente inicializado y listo';
                    }
                } catch (error) {
                    console.log('Could not check system status:', error);
                }
            }
            
            // Check status on load
            document.addEventListener('DOMContentLoaded', checkSystemStatus);
            
            document.getElementById('queryForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const query = document.getElementById('sqlQuery').value;
                const resultsDiv = document.getElementById('results');
                
                // Show loading with special message for first request
                resultsDiv.style.display = 'block';
                
                // Check if system might be initializing
                const healthResponse = await fetch('/health');
                const health = await healthResponse.json();
                
                if (health.status === 'ready' && health.mode === 'lazy') {
                    resultsDiv.innerHTML = '<div class="loading">[INIT] Primera consulta detectada - Inicializando sistema completo (VectorStore/ChromaDB)...<br>[WAIT] Esto puede tomar 1-2 minutos. Por favor espere...</div>';
                } else if (health.status === 'initializing') {
                    resultsDiv.innerHTML = '<div class="loading">[INIT] Sistema aún inicializándose...<br>[WAIT] Por favor espere unos momentos más...</div>';
                } else {
                    resultsDiv.innerHTML = '<div class="loading">[PROCESSING] Analizando query... Esto puede tomar unos momentos.</div>';
                }
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                    
                } catch (error) {
                    console.error('Error:', error);
                    resultsDiv.innerHTML = `<div class="result-section error">
                        <h3>[ERROR] Error</h3>
                        <p>Error al procesar la consulta: ${error.message}</p>
                        <p><small>Si es la primera consulta, el sistema puede estar inicializándose. Intente nuevamente en unos minutos.</small></p>
                    </div>`;
                }
            });
            
            function displayResults(result) {
                const resultsDiv = document.getElementById('results');
                
                let html = '';
                
                // Success/Error status
                if (result.success) {
                    html += `<div class="result-section success">
                        <h3>[SUCCESS] Análisis Completado</h3>
                        <p>Tiempo de procesamiento: ${result.processing_time.toFixed(2)} segundos</p>
                    </div>`;
                } else {
                    html += `<div class="result-section error">
                        <h3>[ERROR] Error en el Análisis</h3>
                        <ul>${result.error_messages.map(err => `<li>${err}</li>`).join('')}</ul>
                    </div>`;
                }
                
                // Standards compliance
                if (result.is_standards_compliant) {
                    html += `<div class="result-section success">
                        <h3>[COMPLIANT] Cumplimiento de Estándares</h3>
                        <p>El query cumple con los estándares de codificación.</p>
                    </div>`;
                } else {
                    html += `<div class="result-section warning">
                        <h3>[VIOLATIONS] Violaciones de Estándares</h3>
                        <ul>${result.standards_violations.map(violation => `<li>${violation}</li>`).join('')}</ul>
                    </div>`;
                }
                
                // Corrected query
                if (result.corrected_query && result.corrected_query !== result.original_query) {
                    html += `<div class="result-section">
                        <h3>[CORRECTED] Query Corregido</h3>
                        <pre>${result.corrected_query}</pre>
                    </div>`;
                }
                
                // Performance assessment
                html += `<div class="result-section">
                    <h3>[PERFORMANCE] Evaluación de Performance</h3>
                    <p><strong>Calificación:</strong> ${result.performance_assessment}</p>
                </div>`;
                
                // Optimization suggestions
                if (result.optimization_suggestions.length > 0) {
                    html += `<div class="result-section">
                        <h3>[OPTIMIZATION] Sugerencias de Optimización</h3>`;
                    
                    result.optimization_suggestions.forEach(suggestion => {
                        html += `<div class="suggestion priority-${suggestion.priority.toLowerCase()}">
                            <h4>[${suggestion.priority}] ${suggestion.issue}</h4>
                            <p><strong>Sugerencia:</strong> ${suggestion.suggestion}</p>
                            <p><strong>Impacto:</strong> ${suggestion.impact}</p>
                            <p><strong>Implementación:</strong> ${suggestion.implementation}</p>
                        </div>`;
                    });
                    
                    html += '</div>';
                }
                
                // EXPLAIN plan
                if (result.explain_plan) {
                    html += `<div class="result-section">
                        <h3>[EXPLAIN] Plan de Ejecución</h3>
                        <pre>${result.explain_plan}</pre>
                    </div>`;
                }
                
                // Final recommendations
                if (result.final_recommendations.length > 0) {
                    html += `<div class="result-section">
                        <h3>[RECOMMENDATIONS] Recomendaciones Finales</h3>
                        <ul>${result.final_recommendations.map(rec => `<li>${rec}</li>`).join('')}</ul>
                    </div>`;
                }
                
                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/analyze", response_model=QueryResponse)
async def analyze_query(request: QueryRequest):
    """Analyze a SQL query using the complete workflow with lazy initialization."""
    start_time = datetime.now()
    query_id = request.query_id or str(uuid.uuid4())
    
    try:
        if not workflow:
            raise HTTPException(status_code=500, detail="Workflow not created")
        
        # Lazy initialization - solo la primera vez
        if not workflow._initialized:
            if workflow._initializing:
                # Si ya se está inicializando, esperar un poco
                raise HTTPException(
                    status_code=503, 
                    detail="Workflow is currently initializing, please retry in a few seconds"
                )
            
            try:
                workflow._initializing = True
                logging.info("[INIT] First request detected - initializing complete workflow...")
                logging.info("[WAIT] This may take 1-2 minutes for VectorStore/ChromaDB setup...")
                
                # Inicializar completamente
                await workflow.initialize()
                workflow._initialized = True
                workflow._initializing = False
                
                logging.info("[SUCCESS] Workflow initialized successfully on first request")
                
            except Exception as e:
                workflow._initializing = False
                logging.error(f"[ERROR] Failed to initialize workflow on demand: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to initialize workflow: {str(e)}"
                )
        
        # Continuar con el procesamiento normal
        result = await workflow.process_query(request.query)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log performance
        perf_logger.log_workflow_performance(
            workflow_id=query_id,
            total_time=processing_time,
            agents_used=["sql_reviewer", "explain_generator", "explain_interpreter"],
            success=result.success,
            error_count=len(result.error_messages)
        )
        
        # Log SQL analysis
        sql_logger.log_query_review(
            query_id=query_id,
            original_query=result.original_query,
            corrected_query=result.corrected_query,
            violations=result.standards_violations,
            is_compliant=result.is_standards_compliant
        )
        
        return QueryResponse(
            query_id=query_id,
            success=result.success,
            original_query=result.original_query,
            corrected_query=result.corrected_query,
            is_standards_compliant=result.is_standards_compliant,
            standards_violations=result.standards_violations,
            explain_plan=result.explain_plan,
            performance_assessment=result.performance_assessment,
            optimization_suggestions=result.optimization_suggestions,
            final_recommendations=result.final_recommendations,
            error_messages=result.error_messages,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Error processing query: {str(e)}"
        
        # Log error
        perf_logger.log_workflow_performance(
            workflow_id=query_id,
            total_time=processing_time,
            agents_used=[],
            success=False,
            error_count=1
        )
        
        logging.error(f"Query analysis failed for {query_id}: {e}")
        
        return QueryResponse(
            query_id=query_id,
            success=False,
            original_query=request.query,
            corrected_query="",
            is_standards_compliant=False,
            standards_violations=[],
            explain_plan="",
            performance_assessment="Error",
            optimization_suggestions=[],
            final_recommendations=[],
            error_messages=[error_msg],
            processing_time=processing_time
        )


@app.get("/health")
async def health_check():
    """Health check endpoint with lazy initialization status."""
    try:
        if workflow:
            if workflow._initialized:
                status = await workflow.get_workflow_status()
                return {
                    "status": "healthy", 
                    "mode": "full",
                    "workflow": status,
                    "initialization": "complete"
                }
            elif workflow._initializing:
                return {
                    "status": "initializing", 
                    "mode": "lazy",
                    "message": "Workflow is currently initializing VectorStore/ChromaDB"
                }
            else:
                return {
                    "status": "ready", 
                    "mode": "lazy",
                    "message": "Workflow ready - will initialize on first request"
                }
        else:
            return {"status": "unhealthy", "error": "Workflow not created"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/examples")
async def get_examples():
    """Get sample SQL queries for testing."""
    examples = [
        {
            "name": "UPDATE con problemas",
            "query": "update empleados set salario=salario*1.1 where departamento='IT' and hire_date<'2020-01-01';"
        },
        {
            "name": "SELECT con JOIN implícito",
            "query": "select * from empleados, departamentos where empleados.dept_id = departamentos.id and salario > 50000 order by salario;"
        },
        {
            "name": "CREATE TABLE básica",
            "query": "create table test (id int, name varchar(100));"
        }
    ]
    return {"examples": examples}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
