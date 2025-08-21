"""
Interfaz Web con Validación Detallada de Agentes
==============================================

Muestra las salidas individuales y detalladas de cada agente:
1. [REVIEWER] SQL Reviewer Agent (RAG + ChromaDB)
2. [EXPLAIN] Explain Generator Agent (MCP + Teradata)  
3. [INTERPRETER] Explain Interpreter Agent (Azure OpenAI)
"""

import asyncio
import logging
import uuid
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.agents.sql_reviewer import SQLReviewerAgent, SQLReviewResult
from src.agents.explain_generator import EnhancedExplainGenerator, EnhancedExplainResult
from src.agents.explain_interpreter import ExplainInterpreterAgent, ExplainAnalysis
from src.rag.vector_store import VectorStore
from src.utils.logging_utils import setup_logging
from src.utils.logging_utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Teradata SQL Agent - Validación Detallada",
    description="Interfaz que muestra salidas detalladas de cada agente",
    version="2.0.0"
)

# Global agents
vector_store: Optional[VectorStore] = None
sql_reviewer: Optional[SQLReviewerAgent] = None
explain_generator: Optional[EnhancedExplainGenerator] = None
explain_interpreter: Optional[ExplainInterpreterAgent] = None

class QueryRequest(BaseModel):
    query: str

class AgentOutput(BaseModel):
    agent_name: str
    success: bool
    processing_time: float
    output_data: Dict[str, Any]
    error_message: Optional[str] = None

class DetailedResponse(BaseModel):
    query_id: str
    original_query: str
    timestamp: str
    total_processing_time: float
    agents_output: List[AgentOutput]
    final_summary: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Inicializar agentes al arrancar la aplicación."""
    global vector_store, sql_reviewer, explain_generator, explain_interpreter
    
    try:
        logger.info("[STARTUP] Inicializando agentes para validación detallada...")
        
        # Initialize vector store
        vector_store = VectorStore()
        await vector_store.initialize()
        doc_count = await vector_store._get_collection_count()
        logger.info(f"[SUCCESS] ChromaDB inicializado con {doc_count} documentos")
        
        # Initialize agents
        sql_reviewer = SQLReviewerAgent(vector_store=vector_store)
        await sql_reviewer.initialize()
        logger.info("[SUCCESS] SQL Reviewer Agent inicializado")
        
        explain_generator = EnhancedExplainGenerator()
        await explain_generator.initialize()
        logger.info("[SUCCESS] Explain Generator Agent inicializado")
        
        explain_interpreter = ExplainInterpreterAgent()
        logger.info("[SUCCESS] Explain Interpreter Agent inicializado")
        
        logger.info("[COMPLETE] Todos los agentes inicializados correctamente")
        
    except Exception as e:
        logger.error(f"[ERROR] Error en inicialización: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al cerrar la aplicación."""
    if explain_generator:
        await explain_generator.cleanup()
    logger.info("[CLEANUP] Cleanup completado")

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Interfaz web detallada."""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Validación Detallada de Agentes - Teradata SQL Agent</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .header-info {
                background: #3498db;
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }
            .query-section {
                margin-bottom: 30px;
            }
            label {
                display: block;
                margin-bottom: 10px;
                font-weight: bold;
                color: #34495e;
                font-size: 1.1em;
            }
            textarea {
                width: 100%;
                height: 200px;
                padding: 15px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                resize: vertical;
                background-color: #f8f9fa;
            }
            textarea:focus {
                outline: none;
                border-color: #3498db;
                background-color: white;
            }
            .analyze-btn {
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                margin-top: 20px;
                width: 100%;
                transition: all 0.3s ease;
            }
            .analyze-btn:hover {
                background: linear-gradient(135deg, #2980b9, #1f4e79);
                transform: translateY(-2px);
            }
            .analyze-btn:disabled {
                background: #95a5a6;
                cursor: not-allowed;
                transform: none;
            }
            .results {
                margin-top: 30px;
                display: none;
            }
            .agent-output {
                background: #f8f9fa;
                border-left: 5px solid #3498db;
                padding: 20px;
                margin: 20px 0;
                border-radius: 0 8px 8px 0;
            }
            .agent-output.success {
                border-left-color: #27ae60;
                background: #f0fff4;
            }
            .agent-output.error {
                border-left-color: #e74c3c;
                background: #fff5f5;
            }
            .agent-header {
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .agent-details {
                background: white;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                border: 1px solid #e1e8ed;
            }
            .detail-item {
                margin: 8px 0;
                padding: 8px 0;
                border-bottom: 1px solid #f0f0f0;
            }
            .detail-item:last-child {
                border-bottom: none;
            }
            .detail-label {
                font-weight: bold;
                color: #2c3e50;
                display: inline-block;
                min-width: 150px;
            }
            .detail-value {
                color: #34495e;
            }
            .code-block {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                overflow-x: auto;
                margin: 10px 0;
                white-space: pre-wrap;
            }
            .suggestions-list {
                list-style: none;
                padding: 0;
            }
            .suggestion-item {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border: 1px solid #e1e8ed;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .suggestion-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            .priority-critical { 
                border-left: 4px solid #e74c3c; 
                background: linear-gradient(to right, #ffebee, white);
            }
            .priority-high { 
                border-left: 4px solid #f39c12; 
                background: linear-gradient(to right, #fff8e1, white);
            }
            .priority-medium { 
                border-left: 4px solid #f1c40f; 
                background: linear-gradient(to right, #fffde7, white);
            }
            .priority-low { 
                border-left: 4px solid #27ae60; 
                background: linear-gradient(to right, #e8f5e8, white);
            }
            .loading {
                text-align: center;
                padding: 40px;
                font-size: 1.2em;
                color: #7f8c8d;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .final-summary {
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 25px;
                margin-top: 30px;
            }
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .summary-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .summary-value {
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }
            .summary-label {
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Validación Detallada de Agentes</h1>
            
            <div class="header-info">
                <h3>Sistema Multi-Agente para Análisis SQL</h3>
                <p><strong>1. SQL Reviewer Agent:</strong> Revisa queries usando RAG con base de conocimiento interna</p>
                <p><strong>2. Explain Generator Agent:</strong> Genera planes EXPLAIN usando MCP Server y Teradata</p>
                <p><strong>3. Explain Interpreter Agent:</strong> Interpreta planes y proporciona sugerencias de optimización</p>
            </div>
            
            <div class="query-section">
                <label for="queryInput">Ingresa tu consulta SQL para análisis detallado:</label>
                <textarea id="queryInput" placeholder="Ejemplo:
SELECT 
    Campo1 AS Alias1,
    Campo2 AS Alias2,
    Campo3 AS Alias3
FROM 
    Otro_Esquema.S_SFS_ACCOUNT
WHERE 
    Campo1 = 'Valor1' 
    AND Campo2 > 100;"></textarea>
                
                <button class="analyze-btn" onclick="analyzeQuery()">Analizar con Validación Detallada</button>
            </div>
            
            <div id="results" class="results">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Procesando query con los 3 agentes...</p>
                    <p><small>Esto puede tomar entre 30-60 segundos</small></p>
                </div>
            </div>
        </div>

        <script>
            async function analyzeQuery() {
                const query = document.getElementById('queryInput').value.trim();
                if (!query) {
                    alert('Por favor ingresa una consulta SQL');
                    return;
                }

                const btn = document.querySelector('.analyze-btn');
                const results = document.getElementById('results');
                
                btn.disabled = true;
                btn.textContent = '⏳ Analizando...';
                results.style.display = 'block';
                results.innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Procesando query con los 3 agentes...</p>
                        <p><small>Esto puede tomar entre 30-60 segundos</small></p>
                    </div>
                `;

                try {
                    const response = await fetch('/analyze-detailed', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({query: query})
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const data = await response.json();
                    displayDetailedResults(data);

                } catch (error) {
                    results.innerHTML = `
                        <div class="agent-output error">
                            <div class="agent-header">[ERROR] Error en el análisis</div>
                            <p><strong>Error:</strong> ${error.message}</p>
                        </div>
                    `;
                } finally {
                    btn.disabled = false;
                    btn.textContent = 'Analizar con Validación Detallada';
                }
            }

            function displayDetailedResults(data) {
                const results = document.getElementById('results');
                
                let html = `
                    <h2>Resultados Detallados de Análisis</h2>
                    <div class="agent-details">
                        <div class="detail-item">
                            <span class="detail-label">Query ID:</span>
                            <span class="detail-value">${data.query_id}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Timestamp:</span>
                            <span class="detail-value">${data.timestamp}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Tiempo Total:</span>
                            <span class="detail-value">${data.total_processing_time.toFixed(2)}s</span>
                        </div>
                    </div>
                    
                    <div class="code-block">
                        <strong>Query Original:</strong><br>
                        ${escapeHtml(data.original_query)}
                    </div>
                `;

                // Agent outputs
                data.agents_output.forEach((agent, index) => {
                    const statusClass = agent.success ? 'success' : 'error';
                    const statusIcon = agent.success ? 'SUCCESS' : 'ERROR';
                    
                    html += `
                        <div class="agent-output ${statusClass}">
                            <div class="agent-header">
                                ${statusIcon} ${agent.agent_name}
                                <span style="font-size: 0.8em; color: #7f8c8d;">(${agent.processing_time.toFixed(2)}s)</span>
                            </div>
                    `;
                    
                    if (agent.error_message) {
                        html += `<div class="agent-details"><p><strong>Error:</strong> ${agent.error_message}</p></div>`;
                    }
                    
                    html += generateAgentDetails(agent.agent_name, agent.output_data);
                    html += `</div>`;
                });

                // Final summary
                html += `
                    <div class="final-summary">
                        <h3>Resumen Final del Análisis</h3>
                        <div class="summary-grid">
                `;

                Object.entries(data.final_summary).forEach(([key, value]) => {
                    html += `
                        <div class="summary-card">
                            <div class="summary-value">${value}</div>
                            <div class="summary-label">${formatSummaryLabel(key)}</div>
                        </div>
                    `;
                });

                html += `
                        </div>
                    </div>
                `;

                results.innerHTML = html;
            }

            function generateAgentDetails(agentName, outputData) {
                let html = '<div class="agent-details">';
                
                if (agentName.includes('SQL Reviewer')) {
                    html += generateReviewerDetails(outputData);
                } else if (agentName.includes('Explain Generator')) {
                    html += generateGeneratorDetails(outputData);
                } else if (agentName.includes('Explain Interpreter')) {
                    html += generateInterpreterDetails(outputData);
                }
                
                html += '</div>';
                return html;
            }

            function generateReviewerDetails(data) {
                let html = `
                    <div class="detail-item">
                        <span class="detail-label">Cumple Estándares:</span>
                        <span class="detail-value">${data.is_compliant ? 'Sí' : 'No'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Confianza:</span>
                        <span class="detail-value">${(data.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                `;
                
                if (data.violations && data.violations.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Violaciones:</span>
                            <div class="detail-value">
                                <ul>
                                    ${data.violations.map(v => `<li>${v}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    `;
                }
                
                if (data.corrected_query) {
                    html += `<div class="code-block"><strong>Query Corregido:</strong><br>${escapeHtml(data.corrected_query)}</div>`;
                }
                
                if (data.recommendations && data.recommendations.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Recomendaciones:</span>
                            <div class="detail-value">
                                <ul>
                                    ${data.recommendations.map(r => `<li>${r}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    `;
                }
                
                return html;
            }

            function generateGeneratorDetails(data) {
                let html = `
                    <div class="detail-item">
                        <span class="detail-label">Usó Fallback:</span>
                        <span class="detail-value">${data.fallback_used ? 'Sí' : 'No'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Timeout:</span>
                        <span class="detail-value">${data.timeout_occurred ? 'Sí' : 'No'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Herramientas MCP:</span>
                        <span class="detail-value">${data.mcp_tools_used ? data.mcp_tools_used.length : 0}</span>
                    </div>
                `;
                
                if (data.explain_plan) {
                    html += `<div class="code-block"><strong>Plan EXPLAIN:</strong><br>${escapeHtml(data.explain_plan.substring(0, 1000))}${data.explain_plan.length > 1000 ? '...' : ''}</div>`;
                }
                
                if (data.warnings && data.warnings.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Advertencias:</span>
                            <div class="detail-value">
                                <ul>
                                    ${data.warnings.map(w => `<li>${w}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    `;
                }
                
                return html;
            }

            function generateInterpreterDetails(data) {
                let html = `
                    <div class="detail-item">
                        <span class="detail-label">Performance:</span>
                        <span class="detail-value">${data.overall_performance}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Complejidad:</span>
                        <span class="detail-value">${data.query_complexity}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Potencial Mejora:</span>
                        <span class="detail-value">${data.estimated_improvement}</span>
                    </div>
                `;
                
                if (data.bottlenecks && data.bottlenecks.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Cuellos de Botella:</span>
                            <div class="detail-value">
                                <ul class="suggestions-list">
                    `;
                    data.bottlenecks.forEach(bottleneck => {
                        // Handle both old string format and new object format
                        if (typeof bottleneck === 'string') {
                            html += `<li>${bottleneck}</li>`;
                        } else {
                            html += `
                                <li class="suggestion-item priority-${bottleneck.priority.toLowerCase()}">
                                    <strong>CRITICAL ${bottleneck.issue}</strong><br>
                                    <em>${bottleneck.description}</em><br>
                                    <small><strong>Prioridad:</strong> ${bottleneck.priority} | <strong>Impacto:</strong> ${bottleneck.impact}</small><br>
                                    <small><strong>Implementación:</strong> ${bottleneck.implementation}</small>
                                </li>
                            `;
                        }
                    });
                    html += `
                                </ul>
                            </div>
                        </div>
                    `;
                }
                
                if (data.suggestions && data.suggestions.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Sugerencias de Optimización:</span>
                            <div class="detail-value">
                                <ul class="suggestions-list">
                    `;
                    data.suggestions.forEach(suggestion => {
                        html += `
                            <li class="suggestion-item priority-${suggestion.priority.toLowerCase()}">
                                <strong>OPTIMIZE ${suggestion.issue}</strong><br>
                                <em>${suggestion.description}</em><br>
                                <small><strong>Prioridad:</strong> ${suggestion.priority} | <strong>Impacto:</strong> ${suggestion.impact}</small><br>
                                <small><strong>Implementación:</strong> ${suggestion.implementation}</small>
                            </li>
                        `;
                    });
                    html += `
                                </ul>
                            </div>
                        </div>
                    `;
                }

                if (data.teradata_specific_notes && data.teradata_specific_notes.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Notas Específicas de Teradata:</span>
                            <div class="detail-value">
                                <ul>
                                    ${data.teradata_specific_notes.map(note => `<li>NOTE ${note}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    `;
                }
                
                return html;
            }

            function formatSummaryLabel(key) {
                const labels = {
                    'total_processing_time': 'Tiempo Total (s)',
                    'standards_compliant': 'Cumple Estándares',
                    'violations_count': 'Violaciones',
                    'suggestions_count': 'Sugerencias',
                    'performance_assessment': 'Assessment',
                    'critical_issues': 'Issues Críticos',
                    'high_priority': 'Alta Prioridad',
                    'medium_priority': 'Media Prioridad',
                    'low_priority': 'Baja Prioridad'
                };
                return labels[key] || key;
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            // Sample queries for testing
            document.addEventListener('DOMContentLoaded', function() {
                const sampleQueries = [
                    `SELECT
    Campo1 AS Alias1,
    Campo2 AS Alias2,
    Campo3 AS Alias3
FROM
    Otro_Esquema.S_SFS_ACCOUNT
WHERE
    Campo1 = 'Valor1'
    AND Campo2 > 100;`,
                    
                    `SELECT * FROM empleados 
WHERE departamento = 'IT' 
AND fecha_ingreso > '2020-01-01';`,
                    
                    `update empleados 
set salario = salario * 1.1 
where departamento = 'IT';`
                ];
                
                // Add sample query buttons
                const container = document.querySelector('.query-section');
                const samplesDiv = document.createElement('div');
                samplesDiv.style.marginTop = '15px';
                samplesDiv.innerHTML = '<p><strong>Queries de ejemplo:</strong></p>';
                
                sampleQueries.forEach((query, index) => {
                    const btn = document.createElement('button');
                    btn.textContent = `Ejemplo ${index + 1}`;
                    btn.style.marginRight = '10px';
                    btn.style.marginBottom = '10px';
                    btn.style.padding = '5px 10px';
                    btn.style.border = '1px solid #3498db';
                    btn.style.borderRadius = '4px';
                    btn.style.background = '#f8f9fa';
                    btn.style.cursor = 'pointer';
                    btn.onclick = () => {
                        document.getElementById('queryInput').value = query;
                    };
                    samplesDiv.appendChild(btn);
                });
                
                container.appendChild(samplesDiv);
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/analyze-detailed", response_model=DetailedResponse)
async def analyze_detailed(request: QueryRequest) -> DetailedResponse:
    """Analiza una query con salidas detalladas de cada agente."""
    global sql_reviewer, explain_generator, explain_interpreter
    
    if not all([sql_reviewer, explain_generator, explain_interpreter]):
        raise HTTPException(status_code=503, detail="Agentes no inicializados")
    
    query_id = str(uuid.uuid4())
    start_time = datetime.now()
    agents_output: List[AgentOutput] = []
    
    try:
        logger.info(f"[ANALYSIS] Iniciando análisis detallado para query {query_id}")
        
        # Agent 1: SQL Reviewer
        agent_start = datetime.now()
        try:
            review_result = await sql_reviewer.review_query(request.query)
            agent_time = (datetime.now() - agent_start).total_seconds()
            
            agents_output.append(AgentOutput(
                agent_name="[REVIEWER] SQL Reviewer Agent",
                success=True,
                processing_time=agent_time,
                output_data={
                    "is_compliant": review_result.is_compliant,
                    "confidence_score": review_result.confidence_score,
                    "violations": review_result.violations,
                    "corrected_query": review_result.corrected_query,
                    "recommendations": review_result.recommendations,
                    "used_examples": len(review_result.used_examples)
                }
            ))
            
            logger.info(f"[SUCCESS] SQL Reviewer completado en {agent_time:.2f}s")
            
        except Exception as e:
            agent_time = (datetime.now() - agent_start).total_seconds()
            agents_output.append(AgentOutput(
                agent_name="[REVIEWER] SQL Reviewer Agent",
                success=False,
                processing_time=agent_time,
                output_data={},
                error_message=str(e)
            ))
            logger.error(f"[ERROR] Error en SQL Reviewer: {e}")
        
        # Agent 2: Explain Generator
        agent_start = datetime.now()
        try:
            # Importante: Usar la query original para el EXPLAIN, no la corregida
            explain_result = await explain_generator.generate_explain_plan(request.query)
            agent_time = (datetime.now() - agent_start).total_seconds()
            
            logger.info(f"[WEB] EXPLAIN result success: {explain_result.success}")
            logger.info(f"[WEB] EXPLAIN plan length: {len(explain_result.explain_plan) if explain_result.explain_plan else 0}")
            logger.info(f"[WEB] EXPLAIN plan preview: {explain_result.explain_plan[:200] if explain_result.explain_plan else 'None'}...")
            
            agents_output.append(AgentOutput(
                agent_name="[EXPLAIN] Explain Generator Agent",
                success=explain_result.success,
                processing_time=agent_time,
                output_data={
                    "success": explain_result.success,
                    "explain_plan": explain_result.explain_plan,
                    "fallback_used": explain_result.fallback_used,
                    "timeout_occurred": explain_result.timeout_occurred,
                    "mcp_tools_used": explain_result.mcp_tools_used,
                    "connection_attempts": explain_result.connection_attempts,
                    "processing_time": explain_result.processing_time,
                    "query_cost": explain_result.query_cost,
                    "execution_time": explain_result.execution_time,
                    "warnings": explain_result.warnings
                }
            ))
            
            logger.info(f"[SUCCESS] Explain Generator completado en {agent_time:.2f}s")
            
        except Exception as e:
            agent_time = (datetime.now() - agent_start).total_seconds()
            agents_output.append(AgentOutput(
                agent_name="[EXPLAIN] Explain Generator Agent",
                success=False,
                processing_time=agent_time,
                output_data={},
                error_message=str(e)
            ))
            logger.error(f"[ERROR] Error en Explain Generator: {e}")
        
        # Agent 3: Explain Interpreter
        agent_start = datetime.now()
        try:
            if 'explain_result' in locals() and explain_result.success:
                analysis_result = await explain_interpreter.analyze_explain_plan(request.query, explain_result)
                agent_time = (datetime.now() - agent_start).total_seconds()
                
                agents_output.append(AgentOutput(
                    agent_name="[INTERPRETER] Explain Interpreter Agent",
                    success=True,
                    processing_time=agent_time,
                    output_data={
                        "overall_performance": analysis_result.overall_performance,
                        "query_complexity": analysis_result.query_complexity,
                        "estimated_improvement": analysis_result.estimated_improvement,
                        "bottlenecks": analysis_result.bottlenecks,
                        "suggestions": [
                            {
                                "issue": suggestion.issue,
                                "description": suggestion.description,
                                "priority": suggestion.priority,
                                "impact": suggestion.impact,
                                "implementation": suggestion.implementation
                            }
                            for suggestion in analysis_result.suggestions
                        ],
                        "teradata_specific_notes": analysis_result.teradata_specific_notes
                    }
                ))
                
                logger.info(f"[SUCCESS] Explain Interpreter completado en {agent_time:.2f}s")
                
            else:
                agent_time = (datetime.now() - agent_start).total_seconds()
                agents_output.append(AgentOutput(
                    agent_name="[INTERPRETER] Explain Interpreter Agent",
                    success=False,
                    processing_time=agent_time,
                    output_data={},
                    error_message="No hay plan EXPLAIN disponible para interpretar"
                ))
                
        except Exception as e:
            agent_time = (datetime.now() - agent_start).total_seconds()
            agents_output.append(AgentOutput(
                agent_name="[INTERPRETER] Explain Interpreter Agent",
                success=False,
                processing_time=agent_time,
                output_data={},
                error_message=str(e)
            ))
            logger.error(f"[ERROR] Error en Explain Interpreter: {e}")
        
        # Calculate final summary
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Extract data for summary
        is_compliant = False
        violations_count = 0
        suggestions_count = 0
        performance_assessment = "Error"
        
        priority_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for agent in agents_output:
            if agent.agent_name.startswith("[REVIEWER]") and agent.success:
                is_compliant = agent.output_data.get("is_compliant", False)
                violations_count = len(agent.output_data.get("violations", []))
            
            if agent.agent_name.startswith("🎯") and agent.success:
                performance_assessment = agent.output_data.get("overall_performance", "Unknown")
                suggestions = agent.output_data.get("suggestions", [])
                suggestions_count = len(suggestions)
                
                for suggestion in suggestions:
                    priority = suggestion.get("priority", "LOW").upper()
                    if priority in priority_counts:
                        priority_counts[priority] += 1
        
        final_summary = {
            "total_processing_time": round(total_time, 2),
            "standards_compliant": "Sí" if is_compliant else "No",
            "violations_count": violations_count,
            "suggestions_count": suggestions_count,
            "performance_assessment": performance_assessment,
            "critical_issues": priority_counts["CRITICAL"],
            "high_priority": priority_counts["HIGH"],
            "medium_priority": priority_counts["MEDIUM"],
            "low_priority": priority_counts["LOW"]
        }
        
        return DetailedResponse(
            query_id=query_id,
            original_query=request.query,
            timestamp=start_time.isoformat(),
            total_processing_time=total_time,
            agents_output=agents_output,
            final_summary=final_summary
        )
        
    except Exception as e:
        logger.error(f"[ERROR] Error general en análisis detallado: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8006)
