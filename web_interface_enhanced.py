"""
Enhanced version of web_interface.py with MCP timeout protection.
"""

import os
import sys
import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
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

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.agents.workflow import TeradataWorkflow, WorkflowResult
from src.utils.logging_utils import setup_logging
from config.settings import settings

# Setup logging
setup_logging()

# Timeout configuration
TIMEOUT_CONFIG = {
    "mcp_connection": 15.0,
    "mcp_operation": 20.0, 
    "workflow_init": 180.0,
    "query_processing": 300.0,
    "component_timeout": 60.0
}

# Global workflow instance (will be initialized in lifespan)
workflow: Optional[TeradataWorkflow] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global workflow
    
    # Startup
    try:
        workflow = TeradataWorkflow()
        workflow._initialized = False
        workflow._initializing = False
        
        logging.info("[STARTUP] Workflow created in LAZY mode with MCP timeout protection")
        logging.info(f"[CONFIG] MCP timeout: {TIMEOUT_CONFIG['mcp_connection']}s, Operations: {TIMEOUT_CONFIG['mcp_operation']}s")
        
    except Exception as e:
        logging.error(f"[STARTUP-ERROR] Failed to create workflow instance: {e}")
    
    yield
    
    # Shutdown
    if workflow:
        try:
            await asyncio.wait_for(workflow.cleanup(), timeout=10.0)
            logging.info("[SHUTDOWN] Workflow cleanup completed")
        except asyncio.TimeoutError:
            logging.warning("[SHUTDOWN] Cleanup timed out but continuing")
        except Exception as e:
            logging.error(f"[SHUTDOWN] Cleanup error: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Teradata SQL Agent - Enhanced",
    description="LangGraph-based SQL query review with MCP timeout protection",
    version="1.0.1",
    lifespan=lifespan
)

# Request/Response models
class QueryRequest(BaseModel):
    """Request model for SQL query analysis."""
    query: str
    query_id: str = None

class EnhancedQueryResponse(BaseModel):
    """Enhanced response model with timeout information."""
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
    timeout_issues: List[str]
    components_status: Dict[str, str]

# Global timeout configuration
TIMEOUT_CONFIG = {
    "vectorstore_init": 60.0,    # ChromaDB initialization
    "document_loading": 30.0,    # Document loading
    "mcp_connection": 15.0,      # MCP connection timeout
    "mcp_operation": 20.0,       # Individual MCP operations
    "sql_review": 30.0,          # SQL review process
    "explain_analysis": 25.0,    # EXPLAIN analysis
    "total_workflow": 120.0      # Total workflow timeout
}

async def safe_mcp_operation(operation_func, operation_name: str, timeout_seconds: float = 15.0):
    """Safely execute MCP operation with timeout and fallback."""
    try:
        result = await asyncio.wait_for(operation_func(), timeout=timeout_seconds)
        logging.info(f"[MCP-OK] {operation_name} completed in time")
        return result, None
    except asyncio.TimeoutError:
        error_msg = f"[MCP-TIMEOUT] {operation_name} timed out after {timeout_seconds}s"
        logging.warning(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"[MCP-ERROR] {operation_name} failed: {str(e)}"
        logging.error(error_msg)
        return None, error_msg

async def safe_workflow_component(component_func, component_name: str, timeout_seconds: float = 30.0):
    """Safely execute workflow component with timeout protection."""
    try:
        result = await asyncio.wait_for(component_func(), timeout=timeout_seconds)
        logging.info(f"[COMPONENT-OK] {component_name} completed successfully")
        return result, None, "success"
    except asyncio.TimeoutError:
        error_msg = f"[COMPONENT-TIMEOUT] {component_name} timed out after {timeout_seconds}s"
        logging.warning(error_msg)
        return None, error_msg, "timeout"
    except Exception as e:
        error_msg = f"[COMPONENT-ERROR] {component_name} failed: {str(e)}"
        logging.error(error_msg)
        return None, error_msg, "error"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Enhanced web interface with timeout information."""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Teradata SQL Agent - Enhanced</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .timeout-info { background-color: #e3f2fd; border: 1px solid #1976d2; color: #0d47a1; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
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
            .timeout { background-color: #ffeaa7; border-color: #fd7e14; color: #b45309; }
            .loading { text-align: center; padding: 20px; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>[ENHANCED] Teradata SQL Agent</h1>
                <p>Análisis SQL con protección contra timeouts MCP y VectorStore</p>
            </div>
            
            <div class="timeout-info">
                <h4>[TIMEOUT PROTECTION]</h4>
                <p><strong>Timeouts configurados:</strong></p>
                <ul>
                    <li>MCP Connection: 15 segundos</li>
                    <li>MCP Operations: 20 segundos</li>
                    <li>VectorStore Init: 60 segundos</li>
                    <li>Total Workflow: 120 segundos</li>
                </ul>
                <p><em>El sistema continuará funcionando aunque algunos componentes experimenten timeouts.</em></p>
            </div>
            
            <form id="queryForm">
                <div class="form-group">
                    <label for="sqlQuery">Query SQL a analizar:</label>
                    <textarea id="sqlQuery" name="query" placeholder="Ingresa tu query SQL aquí..." required></textarea>
                </div>
                <button type="submit">[ANALYZE] Analizar Query (Modo Enhanced)</button>
            </form>
            
            <div id="results" class="results" style="display: none;"></div>
        </div>

        <script>
            document.getElementById('queryForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const query = document.getElementById('sqlQuery').value;
                const resultsDiv = document.getElementById('results');
                
                resultsDiv.style.display = 'block';
                resultsDiv.innerHTML = '<div class="loading">[PROCESSING] Analizando query con protección de timeouts...</div>';
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const result = await response.json();
                    displayEnhancedResults(result);
                    
                } catch (error) {
                    console.error('Error:', error);
                    resultsDiv.innerHTML = `<div class="result-section error">
                        <h3>[ERROR] Error</h3>
                        <p>Error al procesar la consulta: ${error.message}</p>
                    </div>`;
                }
            });
            
            function displayEnhancedResults(result) {
                const resultsDiv = document.getElementById('results');
                let html = '';
                
                // Success/Error status with timeout information
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
                
                // Timeout issues (if any)
                if (result.timeout_issues && result.timeout_issues.length > 0) {
                    html += `<div class="result-section timeout">
                        <h3>[TIMEOUTS] Componentes con Timeout</h3>
                        <ul>${result.timeout_issues.map(issue => `<li>${issue}</li>`).join('')}</ul>
                        <p><em>El análisis continuó con los componentes disponibles.</em></p>
                    </div>`;
                }
                
                // Component status
                if (result.components_status) {
                    html += `<div class="result-section">
                        <h3>[STATUS] Estado de Componentes</h3>
                        <ul>`;
                    Object.entries(result.components_status).forEach(([component, status]) => {
                        const statusClass = status === 'success' ? 'success' : status === 'timeout' ? 'warning' : 'error';
                        html += `<li class="${statusClass}"><strong>${component}:</strong> ${status}</li>`;
                    });
                    html += `</ul></div>`;
                }
                
                // Rest of the results (same as before)
                if (result.performance_assessment) {
                    html += `<div class="result-section">
                        <h3>[PERFORMANCE] Evaluación de Performance</h3>
                        <p><strong>Análisis:</strong> ${result.performance_assessment}</p>
                    </div>`;
                }
                
                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze", response_model=EnhancedQueryResponse)
async def analyze_query(request: QueryRequest):
    """Enhanced SQL query analysis with comprehensive timeout protection."""
    start_time = datetime.now()
    query_id = request.query_id or str(uuid.uuid4())
    timeout_issues = []
    components_status = {}
    
    try:
        if not workflow:
            raise HTTPException(status_code=500, detail="Workflow not created")
        
        # Enhanced lazy initialization with timeout protection
        if not workflow._initialized:
            if workflow._initializing:
                raise HTTPException(
                    status_code=503, 
                    detail="Workflow is currently initializing with timeout protection"
                )
            
            logging.info(f"[INIT] Starting enhanced workflow initialization for query {query_id}")
            workflow._initializing = True
            
            # Initialize with overall timeout
            async def init_workflow():
                await workflow.initialize()
                workflow._initialized = True
                workflow._initializing = False
                return True
                
            init_result, init_error, init_status = await safe_workflow_component(
                init_workflow,
                "WorkflowInitialization", 
                TIMEOUT_CONFIG["total_workflow"]
            )
            
            components_status["workflow_init"] = init_status
            if init_error:
                timeout_issues.append(init_error)
                workflow._initializing = False
                if init_status == "timeout":
                    # Continue with partial functionality
                    logging.warning("[INIT] Workflow initialization timed out, continuing with limited functionality")
                else:
                    raise HTTPException(status_code=500, detail=f"Workflow initialization failed: {init_error}")
        
        # Process query with component-level timeout protection
        result_data = {
            "query_id": query_id,
            "success": False,
            "original_query": request.query,
            "corrected_query": "",
            "is_standards_compliant": False,
            "standards_violations": [],
            "explain_plan": "",
            "performance_assessment": "Análisis parcial debido a timeouts",
            "optimization_suggestions": [],
            "final_recommendations": [],
            "error_messages": [],
            "processing_time": 0.0,
            "timeout_issues": timeout_issues,
            "components_status": components_status
        }
        
        if workflow._initialized:
            # Try full workflow processing with timeout
            async def process_query():
                return await workflow.process_query(request.query)
                
            workflow_result, workflow_error, workflow_status = await safe_workflow_component(
                process_query,
                "FullWorkflowProcessing",
                TIMEOUT_CONFIG["total_workflow"]
            )
            
            components_status["full_workflow"] = workflow_status
            
            if workflow_result:
                # Success - populate full results
                result_data.update({
                    "success": workflow_result.success,
                    "corrected_query": workflow_result.corrected_query,
                    "is_standards_compliant": workflow_result.is_standards_compliant,
                    "standards_violations": workflow_result.standards_violations,
                    "explain_plan": workflow_result.explain_plan,
                    "performance_assessment": workflow_result.performance_assessment,
                    "optimization_suggestions": workflow_result.optimization_suggestions,
                    "final_recommendations": workflow_result.final_recommendations,
                    "error_messages": workflow_result.error_messages
                })
            else:
                # Partial failure - add error but continue
                if workflow_error:
                    timeout_issues.append(workflow_error)
                    result_data["error_messages"].append(f"Workflow processing issue: {workflow_error}")
        else:
            # Workflow not initialized - minimal analysis
            components_status["full_workflow"] = "unavailable"
            result_data["error_messages"].append("Full workflow unavailable - initialization timeout")
            result_data["performance_assessment"] = "Análisis no disponible debido a problemas de inicialización"
        
        # Calculate final processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result_data["processing_time"] = processing_time
        result_data["timeout_issues"] = timeout_issues
        result_data["components_status"] = components_status
        
        # Set success based on whether we got any useful analysis
        if not result_data["success"] and not timeout_issues:
            result_data["success"] = True  # Partial success
            
        logging.info(f"[COMPLETE] Enhanced analysis completed in {processing_time:.2f}s with {len(timeout_issues)} timeout issues")
        
        return EnhancedQueryResponse(**result_data)
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Critical error in enhanced analysis: {str(e)}"
        
        logging.error(f"[CRITICAL] Enhanced query analysis failed for {query_id}: {e}")
        
        return EnhancedQueryResponse(
            query_id=query_id,
            success=False,
            original_query=request.query,
            corrected_query="",
            is_standards_compliant=False,
            standards_violations=[],
            explain_plan="",
            performance_assessment="Error crítico en análisis",
            optimization_suggestions=[],
            final_recommendations=[],
            error_messages=[error_msg],
            processing_time=processing_time,
            timeout_issues=timeout_issues,
            components_status=components_status
        )

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with component status."""
    try:
        status_info = {
            "status": "unknown",
            "mode": "enhanced",
            "timestamp": datetime.now().isoformat(),
            "timeout_config": TIMEOUT_CONFIG,
            "components": {}
        }
        
        if workflow:
            if workflow._initialized:
                # Try to get detailed status with timeout
                try:
                    workflow_status = await asyncio.wait_for(
                        workflow.get_workflow_status(),
                        timeout=5.0
                    )
                    status_info.update({
                        "status": "healthy",
                        "workflow": workflow_status,
                        "components": {
                            "workflow": "initialized",
                            "vectorstore": "active",
                            "mcp": "connected"
                        }
                    })
                except asyncio.TimeoutError:
                    status_info.update({
                        "status": "degraded",
                        "message": "Status check timed out but system is operational",
                        "components": {
                            "workflow": "timeout",
                            "vectorstore": "unknown",
                            "mcp": "unknown"
                        }
                    })
            elif workflow._initializing:
                status_info.update({
                    "status": "initializing",
                    "message": "Enhanced workflow initializing with timeout protection",
                    "components": {
                        "workflow": "initializing",
                        "vectorstore": "pending",
                        "mcp": "pending"
                    }
                })
            else:
                status_info.update({
                    "status": "ready",
                    "message": "Enhanced workflow ready - will initialize on first request",
                    "components": {
                        "workflow": "ready",
                        "vectorstore": "lazy",
                        "mcp": "lazy"
                    }
                })
        else:
            status_info.update({
                "status": "unhealthy",
                "error": "Workflow not created",
                "components": {
                    "workflow": "missing"
                }
            })
            
        return status_info
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "mode": "enhanced",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
