"""
Explain Interpreter Agent that analyzes EXPLAIN plans and provides optimization suggestions.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.explain_generator import EnhancedExplainResult
from src.utils.azure_openai_utils import azure_openai_manager
from config.settings import settings


class OptimizationPriority(Enum):
    """Priority levels for optimization suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OptimizationSuggestion(BaseModel):
    """Individual optimization suggestion."""
    issue: str = Field(description="Description of the performance issue")
    description: str = Field(description="Specific optimization suggestion")
    priority: str = Field(description="Priority level of the optimization")
    impact: str = Field(description="Expected impact of the optimization")
    implementation: str = Field(description="How to implement the suggestion")


class ExplainAnalysis(BaseModel):
    """Complete analysis of an EXPLAIN plan."""
    overall_performance: str = Field(description="Overall performance assessment")
    bottlenecks: List[str] = Field(description="Identified performance bottlenecks")
    suggestions: List[OptimizationSuggestion] = Field(description="Optimization suggestions")
    query_complexity: str = Field(description="Complexity level of the query")
    estimated_improvement: str = Field(description="Estimated improvement potential")
    teradata_specific_notes: List[str] = Field(description="Teradata-specific observations")


@dataclass
class ExplainMetrics:
    """Metrics extracted from EXPLAIN plan."""
    table_scans: int = 0
    index_scans: int = 0
    joins: int = 0
    sorts: int = 0
    redistributions: int = 0
    spool_operations: int = 0
    product_joins: int = 0
    confidence_level: str = "Unknown"
    estimated_rows: Optional[int] = None
    estimated_cost: Optional[float] = None


class ExplainInterpreterAgent:
    """Agent that interprets EXPLAIN plans and provides optimization suggestions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = azure_openai_manager.get_llm()
        
        # Create the analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt())
        ])
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for EXPLAIN analysis."""
        return """Eres un experto en optimización de queries SQL para Teradata con más de 15 años de experiencia en análisis de planes de ejecución y tuning de performance.

EXPERTISE:
- Análisis profundo de planes EXPLAIN de Teradata
- Optimización de queries complejas en entornos de data warehousing
- Conocimiento avanzado de arquitectura MPP (Massively Parallel Processing)
- Experiencia en identificación de cuellos de botella y optimizaciones

CONTEXTO TERADATA:
- Arquitectura shared-nothing con AMP (Access Module Processors)
- Importancia de la distribución de datos (Primary Index)
- Operaciones de redistribución y duplicación
- Uso de estadísticas y optimizador basado en costos
- Spool space y gestión de memoria

TIPOS DE PROBLEMAS COMUNES:
1. **Product Joins**: Joins cartesianos que causan explosión de datos
2. **Redistributions**: Movimiento excesivo de datos entre AMPs
3. **Table Scans**: Lectura completa de tablas grandes sin índices
4. **Skewed Data**: Distribución desigual de datos
5. **Missing Statistics**: Estadísticas faltantes o desactualizadas
6. **Sort Operations**: Ordenamientos costosos en memoria
7. **Spool Overflow**: Desbordamiento de espacio temporal

NIVELES DE PRIORIDAD:
- **CRITICAL**: Problemas que pueden causar fallos o timeouts
- **HIGH**: Impacto significativo en performance (>50% mejora esperada)
- **MEDIUM**: Mejoras moderadas (10-50% mejora esperada)
- **LOW**: Optimizaciones menores (<10% mejora esperada)

FORMATO DE SALIDA:
Proporciona un análisis estructurado en formato JSON que incluya:
- Evaluación general de performance
- Lista de cuellos de botella identificados
- Sugerencias específicas con prioridad e impacto
- Nivel de complejidad del query
- Potencial de mejora estimado
- Notas específicas de Teradata

Sé específico y práctico en tus recomendaciones."""

    def _get_human_prompt(self) -> str:
        """Get the human prompt template."""
        return """Analiza el siguiente plan EXPLAIN de Teradata:

QUERY ORIGINAL:
```sql
{original_query}
```

PLAN EXPLAIN:
```
{explain_plan}
```

MÉTRICAS EXTRAÍDAS:
- Table Scans: {table_scans}
- Index Scans: {index_scans}
- Joins: {joins}
- Sorts: {sorts}
- Redistributions: {redistributions}
- Spool Operations: {spool_operations}
- Product Joins: {product_joins}
- Confidence Level: {confidence_level}
- Estimated Cost: {estimated_cost}

WARNINGS DEL SISTEMA:
{warnings}

Proporciona un análisis completo que incluya:

1. **Evaluación General**: Califica el performance general (Excelente/Bueno/Regular/Malo/Crítico)

2. **Cuellos de Botella**: Identifica los principales problemas de performance

3. **Sugerencias de Optimización**: Para cada problema, proporciona:
   - Descripción del issue
   - Sugerencia específica de optimización
   - Prioridad (CRITICAL/HIGH/MEDIUM/LOW)
   - Impacto esperado
   - Pasos de implementación

4. **Complejidad**: Evalúa la complejidad del query (Simple/Moderada/Compleja/Muy Compleja)

5. **Potencial de Mejora**: Estima el potencial de mejora (Bajo/Medio/Alto/Muy Alto)

6. **Notas Específicas de Teradata**: Observaciones específicas de la plataforma

Responde ÚNICAMENTE con JSON válido siguiendo el formato especificado."""

    async def analyze_explain(self, explain_plan: str, original_query: str) -> ExplainAnalysis:
        """
        Simplified method to analyze an EXPLAIN plan.
        Compatible with the test flow and MCP responses.
        """
        try:
            self.logger.info("Starting EXPLAIN plan analysis")
            
            # Check if this is an error response from MCP
            if self._is_error_response(explain_plan):
                return self._create_connectivity_error_analysis(explain_plan)
            
            # Extract metrics from explain plan
            metrics = self._extract_metrics(explain_plan)
            
            # Format the prompt
            formatted_prompt = self.analysis_prompt.format_messages(
                original_query=original_query,
                explain_plan=explain_plan,
                table_scans=metrics.table_scans,
                index_scans=metrics.index_scans,
                joins=metrics.joins,
                sorts=metrics.sorts,
                redistributions=metrics.redistributions,
                spool_operations=metrics.spool_operations,
                product_joins=metrics.product_joins,
                confidence_level=metrics.confidence_level,
                estimated_cost=metrics.estimated_cost or "Unknown",
                warnings="None"
            )
            
            # Get LLM response
            response = await self.llm.ainvoke(formatted_prompt)
            
            # Parse JSON response
            import json
            try:
                analysis_dict = json.loads(response.content)
                # Convert to Pydantic model
                analysis = ExplainAnalysis(**analysis_dict)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
                return self._create_fallback_analysis(explain_plan, metrics)
            
            self.logger.info(f"Analysis completed: {analysis.overall_performance}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing EXPLAIN plan: {e}")
            return self._create_error_analysis(str(e))

    async def analyze_explain_plan(self, original_query: str, explain_result: EnhancedExplainResult) -> ExplainAnalysis:
        """Analyze an EXPLAIN plan and provide optimization suggestions."""
        try:
            self.logger.info("Starting EXPLAIN plan analysis from EnhancedExplainResult")
            
            if not explain_result.success:
                return self._create_error_analysis(explain_result.error_message)
            
            # Use the simplified analyze_explain method
            return await self.analyze_explain(explain_result.explain_plan, original_query)
            
        except Exception as e:
            self.logger.error(f"Error analyzing EXPLAIN plan: {e}")
            return self._create_error_analysis(str(e))
    
    def _is_error_response(self, explain_plan: str) -> bool:
        """Check if the explain plan is actually an error response from MCP."""
        error_indicators = [
            "connection error",
            "failed to connect",
            "timeout",
            "dial tcp",
            "i/o timeout",
            "error executing query"
        ]
        plan_lower = explain_plan.lower()
        return any(indicator in plan_lower for indicator in error_indicators)
    
    def _create_connectivity_error_analysis(self, error_response: str) -> ExplainAnalysis:
        """Create analysis for connectivity errors from MCP."""
        return ExplainAnalysis(
            overall_performance="Crítico - Error de Conectividad",
            bottlenecks=[
                "No se pudo conectar a Teradata",
                "Error de red o firewall",
                "Servidor de base de datos inaccesible"
            ],
            suggestions=[
                OptimizationSuggestion(
                    issue="Conectividad a Teradata fallida",
                    description="Verificar conectividad de red y configuración de firewall",
                    priority="CRITICAL",
                    impact="Impide la ejecución de queries",
                    implementation="Contactar administrador de red para habilitar acceso al puerto 1025 de Teradata"
                ),
                OptimizationSuggestion(
                    issue="Configuración de red corporativa",
                    description="Desplegar aplicación en ambiente con acceso directo a Teradata",
                    priority="HIGH",
                    impact="Permitirá análisis completo de performance",
                    implementation="Migrar a servidor dentro de la red corporativa o configurar VPN"
                )
            ],
            query_complexity="No evaluable",
            estimated_improvement="No evaluable sin conectividad",
            teradata_specific_notes=[
                "Error de timeout en puerto 1025 (puerto estándar Teradata)",
                "La aplicación está intentando conexión real usando teradatasql",
                "Infraestructura técnica correcta, problema de conectividad de red"
            ]
        )
    
    def _create_fallback_analysis(self, explain_plan: str, metrics: ExplainMetrics) -> ExplainAnalysis:
        """Create a fallback analysis based on extracted metrics when LLM fails."""
        # Basic analysis based on metrics
        issues = []
        suggestions = []
        
        if metrics.product_joins > 0:
            issues.append("Product joins detectados")
            suggestions.append(OptimizationSuggestion(
                issue="Product join detectado",
                description="Revisar condiciones de JOIN para evitar productos cartesianos",
                priority="CRITICAL",
                impact="Reducción significativa en tiempo de ejecución",
                implementation="Agregar condiciones WHERE apropiadas en los JOINs"
            ))
        
        if metrics.table_scans > 3:
            issues.append("Múltiples table scans")
            suggestions.append(OptimizationSuggestion(
                issue="Exceso de table scans",
                description="Considerar agregar índices en columnas frecuentemente consultadas",
                priority="HIGH",
                impact="Mejora en velocidad de acceso a datos",
                implementation="Crear índices secundarios o ajustar Primary Index"
            ))
        
        if metrics.redistributions > 5:
            issues.append("Redistribución excesiva de datos")
            suggestions.append(OptimizationSuggestion(
                issue="Múltiples redistribuciones",
                description="Optimizar estrategia de JOIN y distribución de datos",
                priority="MEDIUM",
                impact="Reducción en movimiento de datos entre AMPs",
                implementation="Revisar Primary Index design y orden de JOINs"
            ))
        
        # Performance assessment
        if metrics.product_joins > 0:
            performance = "Crítico"
        elif len(issues) > 2:
            performance = "Malo"
        elif len(issues) > 0:
            performance = "Regular"
        else:
            performance = "Bueno"
        
        return ExplainAnalysis(
            overall_performance=performance,
            bottlenecks=issues if issues else ["Sin problemas críticos detectados"],
            suggestions=suggestions if suggestions else [
                OptimizationSuggestion(
                    issue="Query appears optimized",
                    description="No se detectaron problemas críticos de performance",
                    priority="LOW",
                    impact="Mantener configuración actual",
                    implementation="Monitorear performance en ejecución real"
                )
            ],
            query_complexity="Moderada" if metrics.joins > 2 else "Simple",
            estimated_improvement="Alto" if len(issues) > 2 else "Bajo",
            teradata_specific_notes=[
                f"Table scans: {metrics.table_scans}",
                f"Joins: {metrics.joins}",
                f"Redistributions: {metrics.redistributions}",
                f"Confidence level: {metrics.confidence_level}"
            ]
        )
    
    def _extract_metrics(self, explain_plan: str) -> ExplainMetrics:
        """Extract metrics from the EXPLAIN plan text."""
        metrics = ExplainMetrics()
        
        try:
            plan_lower = explain_plan.lower()
            
            # Count operations
            metrics.table_scans = len(re.findall(r'table scan|full table scan|all-rows scan', plan_lower))
            metrics.index_scans = len(re.findall(r'index scan|unique index scan|nusi.*scan|primary index', plan_lower))
            metrics.joins = len(re.findall(r'join|merge join|hash join|nested loop', plan_lower))
            metrics.sorts = len(re.findall(r'sort', plan_lower))
            metrics.redistributions = len(re.findall(r'redistribute|redistribution', plan_lower))
            metrics.spool_operations = len(re.findall(r'spool', plan_lower))
            metrics.product_joins = len(re.findall(r'product join|cartesian', plan_lower))
            
            # Extract confidence level
            if 'high confidence' in plan_lower:
                metrics.confidence_level = 'High'
            elif 'low confidence' in plan_lower:
                metrics.confidence_level = 'Low'
            elif 'no confidence' in plan_lower:
                metrics.confidence_level = 'None'
            
            # Try to extract estimated rows and cost
            cost_match = re.search(r'cost[:\s]*(\d+(?:\.\d+)?)', plan_lower)
            if cost_match:
                metrics.estimated_cost = float(cost_match.group(1))
                
            rows_match = re.search(r'estimated.*?(\d+)\s*rows?', plan_lower)
            if rows_match:
                metrics.estimated_rows = int(rows_match.group(1))
                
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
            
        return metrics
    
    def _create_error_analysis(self, error_message: str) -> ExplainAnalysis:
        """Create an error analysis when explain plan analysis fails."""
        return ExplainAnalysis(
            overall_performance="Error",
            bottlenecks=[f"Analysis failed: {error_message}"],
            suggestions=[
                OptimizationSuggestion(
                    issue="EXPLAIN plan analysis failed",
                    description="Review query syntax and database connection",
                    priority="CRITICAL",
                    impact="Cannot provide optimization without valid EXPLAIN plan",
                    implementation="Fix the underlying issue preventing EXPLAIN generation"
                )
            ],
            query_complexity="Unknown",
            estimated_improvement="Unknown",
            teradata_specific_notes=["EXPLAIN plan could not be generated or analyzed"]
        )
    
    async def get_quick_assessment(self, explain_result: EnhancedExplainResult) -> Dict[str, Any]:
        """Get a quick performance assessment without full LLM analysis."""
        if not explain_result.success:
            return {
                "status": "error",
                "message": explain_result.error_message,
                "recommendations": ["Fix query syntax or connection issues"]
            }
        
        metrics = self._extract_metrics(explain_result.explain_plan)
        
        # Quick assessment based on metrics
        issues = []
        if metrics.product_joins > 0:
            issues.append("Product joins detected - high performance risk")
        if metrics.table_scans > 3:
            issues.append("Multiple table scans - consider adding indexes")
        if metrics.redistributions > 5:
            issues.append("Excessive data redistribution - review join conditions")
        if metrics.confidence_level == "Low" or metrics.confidence_level == "None":
            issues.append("Low optimizer confidence - update statistics")
        
        # Overall status
        if metrics.product_joins > 0 or metrics.confidence_level == "None":
            status = "critical"
        elif metrics.table_scans > 3 or metrics.redistributions > 5:
            status = "warning"
        elif len(issues) > 0:
            status = "attention"
        else:
            status = "good"
        
        return {
            "status": status,
            "metrics": {
                "table_scans": metrics.table_scans,
                "index_scans": metrics.index_scans,
                "joins": metrics.joins,
                "redistributions": metrics.redistributions,
                "product_joins": metrics.product_joins,
                "confidence": metrics.confidence_level
            },
            "issues": issues,
            "recommendations": self._get_quick_recommendations(metrics)
        }
    
    def _get_quick_recommendations(self, metrics: ExplainMetrics) -> List[str]:
        """Get quick recommendations based on metrics."""
        recommendations = []
        
        if metrics.product_joins > 0:
            recommendations.append("Add proper join conditions to eliminate product joins")
        
        if metrics.table_scans > 2:
            recommendations.append("Consider adding indexes on frequently queried columns")
        
        if metrics.redistributions > 3:
            recommendations.append("Review Primary Index design and join strategies")
        
        if metrics.confidence_level in ["Low", "None"]:
            recommendations.append("Update table statistics with COLLECT STATISTICS")
        
        if metrics.sorts > 2:
            recommendations.append("Consider adding ORDER BY optimization or pre-sorted tables")
        
        if not recommendations:
            recommendations.append("Query appears to be well-optimized")
        
        return recommendations
    
    async def compare_explain_plans(self, original_analysis: ExplainAnalysis, 
                                   optimized_analysis: ExplainAnalysis) -> Dict[str, Any]:
        """Compare two EXPLAIN plan analyses."""
        try:
            # Extract numeric metrics for comparison
            def extract_priority_count(analysis: ExplainAnalysis) -> Dict[str, int]:
                counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
                for suggestion in analysis.suggestions:
                    priority = suggestion.priority.upper()
                    if priority in counts:
                        counts[priority] += 1
                return counts
            
            original_priorities = extract_priority_count(original_analysis)
            optimized_priorities = extract_priority_count(optimized_analysis)
            
            # Calculate improvements
            improvements = {}
            for priority in original_priorities:
                improvement = original_priorities[priority] - optimized_priorities[priority]
                if improvement != 0:
                    improvements[f"{priority.lower()}_issues"] = improvement
            
            return {
                "improvement_summary": improvements,
                "original_bottlenecks": len(original_analysis.bottlenecks),
                "optimized_bottlenecks": len(optimized_analysis.bottlenecks),
                "performance_change": {
                    "from": original_analysis.overall_performance,
                    "to": optimized_analysis.overall_performance
                },
                "complexity_change": {
                    "from": original_analysis.query_complexity,
                    "to": optimized_analysis.query_complexity
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing EXPLAIN plans: {e}")
            return {"error": str(e)}
