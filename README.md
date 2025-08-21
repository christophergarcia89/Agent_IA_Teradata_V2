# 🔍 Teradata SQL Agent

Sistema inteligente multi-agente de análisis y optimización de queries SQL para Teradata basado en LangGraph, LCEL y técnicas RAG con conectividad real validada.

## 📋 Descripción

Este proyecto implementa un sistema multi-agente productivo que utiliza LangGraph para orquestar tres agentes especializados con conectividad real a Teradata:

1. **🔍 SQL Reviewer Agent**: Revisa queries usando RAG con base de conocimiento interna de estándares
2. **📊 Explain Generator Agent**: Genera planes EXPLAIN usando MCP Server con conexión real a Teradata
3. **🎯 Explain Interpreter Agent**: Interpreta planes y proporciona sugerencias de optimización específicas

## 🏗️ Arquitectura

```
┌─────────────────────┐    ┌───────────────────┐     ┌─────────────────────┐
│   SQL Reviewer      │───▶│ Explain Generator │───▶│ Explain Interpreter │
│    (RAG)            │    │    (MCP Server)   │     │   (Optimization)    │
│Corporate Vector Store│    │     Teradata      │     │  Azure OpenAI       │
└─────────────────────┘    └───────────────────┘     └─────────────────────┘
         │                           │                        │
         ▼                           ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                               │
│                     (LCEL Orchestration)                            │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Teradata Database (IP: 10.33.84.36)                    │
│         Conectividad Validada - Versión 17.20.03.28                 │
└─────────────────────────────────────────────────────────────────────┘

🏢 Corporate Vector Store Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│ Local File System Storage (ChromaDB Replacement)                    │
│ ├── documents.json (4,550+ SQL examples)                           │
│ ├── embeddings.npy (768-dim Jina embeddings)                       │
│ ├── metadata.json (structured metadata)                            │
│ └── stats.json (performance metrics)                               │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Características

- ✅ **Conectividad Real Validada** con Teradata Database (IP: 10.33.84.36)
- 🏢 **Solución Corporativa** con Vector Store local que reemplaza ChromaDB para entornos restrictivos
- 🔍 **Revisión de estándares** usando base de conocimiento interna con 4,550+ ejemplos SQL
- 📊 **Generación de planes EXPLAIN** con MCP Server y teradatasql library
- 🎯 **Análisis inteligente** de performance con sugerencias específicas
- 🔄 **Workflow automatizado** con LangGraph y circuit breaker patterns
- 🌐 **Interfaz web enhanced** con protección de timeout y fallback
- 📝 **Logging completo** con métricas detalladas de performance
- 🏗️ **Arquitectura modular** con manejo robusto de errores
- 🔐 **Timeout Protection** y circuit breaker para conexiones MCP
- 🎛️ **Modo Fallback** para operaciones sin MCP Server
- ⚡ **Inicialización ultra-rápida** (< 2 segundos) con carga progresiva inteligente
- 🛡️ **SSL Bypass automático** para certificados corporativos
- 💾 **Persistencia local** con almacenamiento en archivos JSON/NumPy
- 🔧 **Validadores independientes** para ChromaDB y conectividad MCP
- 🌐 **Interfaz web enhanced** con validación detallada de agentes
- 🔍 **Auto-carga de documentos** cuando la base vectorial está vacía
- ⚡ **Warm-up de embeddings** para resultados de búsqueda consistentes
- 📊 **Métricas detalladas** de performance y timeout tracking

## 📁 Estructura del Proyecto

```
Agent_IA_Teradata/
├── src/
│   ├── agents/
│   │   ├── sql_reviewer.py              # Agente revisor con RAG + timeout safety
│   │   ├── explain_generator.py         # Agente generador EXPLAIN + circuit breaker
│   │   ├── explain_interpreter.py       # Agente intérprete
│   │   └── workflow.py                  # Orquestación LangGraph
│   ├── rag/
│   │   ├── document_loader.py           # Carga de documentos
│   │   └── vector_store.py              # Vector Store corporativo con warm-up
│   ├── utils/
│   │   ├── teradata_utils.py            # Utilidades Teradata
│   │   ├── enhanced_azure_openai_utils.py # Utils Azure OpenAI
│   │   └── logging_utils.py             # Sistema de logging
│   └── web_interface_enhanced.py        # Interfaz con timeout protection
├── knowledge_base/
│   ├── documentation/                   # Documentación de estándares
│   │   ├── estandares_sql_teradata.txt
│   │   ├── mejores_practicas_performance.txt
│   │   └── Normativa.txt
│   └── examples/                        # Ejemplos consolidados OK/NOK
│       ├── consolidado_CREATE_UPDATE.txt
│       ├── consolidado_CREATE-SELECT-INSERT.txt
│       └── consolidado_UPDATE.txt
├── data/
│   └── chroma_db/                       # Base de datos vectorial ChromaDB
├── config/
│   ├── settings.py                      # Configuración centralizada
│   └── model_strategy.py               # Estrategia de modelos
├── test/
│   ├── test_mcp_validation.py           # ✅ Test validación MCP
│   ├── test_azure_connection.py         # Test Azure OpenAI
│   ├── validate_chromadb.py             # 🆕 Validador independiente ChromaDB
│   ├── test_rag_system_fixed.py         # Test sistema RAG mejorado
├── logs/                                # Directorio de logs
│   ├── teradata_agent.log
│   ├── sql_queries.log
│   ├── performance.log
│   └── errors.log
├── mcp_server.py                        # ✅ MCP Server
├── main.py                              # Punto de entrada principal
├── web_interface_validation.py          # 🆕 Interfaz web con validación detallada
├── requirements.txt                     # Dependencias Python (MCP incluidas)
├── .env                                 # ✅ Configuración validada
└── README.md                            # Esta documentación
```

## 🛠️ Instalación y Configuración

### 1. Prerrequisitos

- Python 3.11 o superior
- Acceso a Azure OpenAI
- Conectividad a red donde está Teradata (IP: 10.33.84.36)
- Credenciales válidas de Teradata

### 2. Clonar e instalar

```bash
git clone <repository-url>
cd Agent_IA_Teradata_V2

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Crear directorios necesarios

```bash
# Crear carpetas para datos y logs
mkdir data
mkdir logs
mkdir data\chroma_db

# En Linux/macOS
# mkdir -p data/chroma_db logs
```

### 4. Configurar variables de entorno

### Solicitar configuración de las variables de entorno.

### 5. Verificar conectividad

```bash
# Test de validación MCP
python test\test_mcp_validation.py
```

## 🏃‍♂️ Uso

### 1. Iniciar MCP Server

```bash
# Terminal 1: Iniciar MCP Server
python mcp_server.py
```

Deberías ver:
```
✅ teradatasql library available
🚀 Iniciando Teradata MCP Server (Real Connection)...
📍 URL: http://localhost:3002
📚 Documentación: http://localhost:3002/docs
INFO:     Uvicorn running on http://0.0.0.0:3002
```

### 2. Interfaz Web (Recomendado)

```bash
# Terminal 2: Iniciar interfaz web
python web_interface_enhanced.py
```

Abre tu navegador en: `http://localhost:8006`

### 4. Interfaz Web Enhanced con Validación Detallada (Nuevo)

```bash
# Terminal: Interfaz web con validación de agentes individuales
python web_interface_validation.py
```

Abre tu navegador en: `http://localhost:8006`

**Características de la nueva interfaz:**
- Muestra salidas detalladas de cada agente por separado
- Tracking de tiempo de procesamiento individual
- Visualización de errores y timeouts específicos
- Validación en tiempo real de conectividad MCP y ChromaDB

### 5. Validadores Independientes

```bash
# Validador completo de ChromaDB con auto-carga
python test\validate_chromadb.py

# Test completo del sistema RAG
python test\test_rag_system_fixed.py
```

## 🧪 Testing y Validación

### Tests de Conectividad Validados

```bash
# ✅ Test validación MCP completa (VALIDADO)
python test\test_mcp_validation.py

# Resultado esperado:
# ✅ EXPLAIN exitoso en X.XX segundos
# 🔧 Tools MCP usadas: 1
# 📄 Plan preview: -> The row is sent directly back...
# 🎉 RESULTADO: ✅ MCP VALIDADO EXITOSAMENTE
```

### Tests Adicionales y Nuevos Validadores

```bash
# ✅ Test validación independiente ChromaDB
python test\validate_chromadb.py

# Test Azure OpenAI
python test\test_azure_connection.py

# Test RAG system mejorado
python test\test_rag_system_fixed.py

# Interfaz web con validación detallada
python web_interface_validation.py
```

## 🔧 Configuración Técnica Validada

### 🔧 Configuración Técnica Validada

### Conectividad Teradata Confirmada

- **✅ IP Funcional**: `10.33.84.36:1025`
- **✅ Usuario**: `Usr_Mkt_Common`
- **✅ Database**: `EDW`
- **✅ Versión Teradata**: `17.20.03.28`
- **✅ Driver teradatasql**: `20.0.0.33`

### Configuraciones de Red Validadas

```
❌ IP Bloqueada: 161.131.180.193 (Firewall)
❌ Hostname Bloqueado: EDW (DNS/Firewall)
✅ IP Funcional: 10.33.84.36 (Conectividad confirmada)
```

### MCP Server Optimizado

- **✅ Parser URI corregido**: Maneja correctamente `teradata://` (11 caracteres)
- **✅ Host sin puerto**: Usa solo IP para conexión
- **✅ Timeout apropiado**: 10 segundos (10000ms)
- **✅ Manejo de EXPLAIN**: Procesa correctamente respuestas `{"Explanation": "text"}`
- **✅ Circuit breaker**: Protección automática contra timeouts
- **✅ Enhanced logging**: Tracking detallado de operaciones y errores

### Sistema RAG Mejorado

- **✅ Warm-up automático**: Modelo de embeddings pre-inicializado
- **✅ Auto-carga documentos**: Carga automática si la base está vacía
- **✅ Búsqueda consistente**: Resultados consistentes en múltiples ejecuciones
- **✅ Validador independiente**: Diagnóstico completo con `validate_chromadb.py`
- **✅ Timeout protection**: Búsquedas RAG con protección de timeout

## 📊 Ejemplos de Uso Validados

## 📊 Ejemplos de Uso Validados

### Ejemplo 1: EXPLAIN Query Simple

**Input:**
```sql
SELECT USER, DATABASE, SESSION
```

**✅ Output MCP Validado:**
```json
{
  "success": true,
  "plan": "-> The row is sent directly back to the user as the result of statement 1.",
  "processing_time": 2.34,
  "tools_used": 1,
  "teradata_version": "17.20.03.28"
}
```

### Ejemplo 2: Validación ChromaDB con Auto-carga

**Command:**
```bash
python test\validate_chromadb.py
```

**✅ Output Esperado:**
```
[SUCCESS] VectorStore inicializado en 1.23 segundos
[INFO] Documentos encontrados en ChromaDB: 4550
[SUCCESS] Operaciones de búsqueda funcionando correctamente
[SUCCESS] CHROMADB COMPLETAMENTE OPERATIVO
```

### Ejemplo 3: Interfaz Web con Validación Detallada

**URL:** `http://localhost:8006` (después de ejecutar `python web_interface_validation.py`)

**Características:**
- Muestra salidas individuales de cada agente
- Tracking de tiempo de procesamiento
- Validación en tiempo real de conectividad
- Manejo de timeouts y errores


TERADATA_PASSWORD=tu_password
TERADATA_DATABASE=tu_database

# MCP Server
TERADATA_MCP_SERVER_URL=http://localhost:3002
```

### 5. Preparar base de conocimiento

Agrega tus archivos de documentación y ejemplos:

- Documentación en `knowledge_base/documentation/` (archivos .txt)
- Ejemplos SQL en `knowledge_base/examples/` (archivos .sql)
  - Formato: `CATEGORIA - Ejemplo X OK.sql` / `CATEGORIA - Ejemplo X NOK.sql`

## 🏃‍♂️ Uso

### Interfaz Web (Recomendado)

```bash
python src/web_interface.py
```

Abre tu navegador en: `http://localhost:8000`

### Línea de Comandos

```bash
python main.py
```

### API REST

```bash
curl -X POST "http://localhost:8006/analyze" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "SELECT * FROM empleados WHERE dept = \\"IT\\""}'
```

## 🎯 Ejemplos de Uso

### Ejemplo 1: Query con problemas de estándares

**Input:**
```sql
update empleados set salario=salario*1.1 where departamento='IT' and hire_date<'2020-01-01';
```

**Output:**
- ❌ Violaciones detectadas: nomenclatura, formato, transacciones
- 🔧 Query corregido con mejores prácticas
- 📊 Plan EXPLAIN y análisis de performance
- 🎯 Sugerencias específicas de optimización

### Ejemplo 2: SELECT con JOIN implícito

**Input:**
```sql
select * from empleados, departamentos where empleados.dept_id = departamentos.id;
```

**Output:**
- ⚠️ JOIN implícito detectado
- 🔧 Conversión a INNER JOIN explícito
- 📊 Análisis de distribución de datos
- 🎯 Recomendaciones de índices

## 🔧 Configuración Avanzada

### Personalizar prompts

Edita los prompts en:
- `src/agents/sql_reviewer.py` - Prompt del revisor
- `src/agents/explain_interpreter.py` - Prompt del intérprete

### Agregar nuevos estándares

1. Agrega documentación en `knowledge_base/documentation/`
2. Agrega ejemplos OK/NOK en `knowledge_base/examples/`
3. Reinicia para recargar la base de conocimiento

### Configurar logging

Personaliza en `src/utils/logging_utils.py`:
- Niveles de log
- Formatos de salida
- Rotación de archivos

## 📊 Monitoreo y Métricas

El sistema genera logs detallados en:

- `logs/teradata_agent.log` - Log general
- `logs/sql_queries.log` - Queries analizados
- `logs/performance.log` - Métricas de performance
- `logs/errors.log` - Errores del sistema

### Métricas disponibles:

- Tiempo de procesamiento por agente
- Cantidad de violaciones detectadas
- Éxito/fallo de generación EXPLAIN
- Distribución de prioridades en sugerencias

## 🔌 Integración con MCP Server

Para usar el teradata-mcp-server:

1. Instala y configura teradata-mcp-server
2. Inicia el servidor en puerto 3002
3. Configura `TERADATA_MCP_SERVER_URL` en .env

Ejemplo de configuración MCP:

```json
{
  "host": "localhost",
  "port": 3002,
  "database": "DBC",
  "timeout": 30
}
```

## 🚨 Troubleshooting

### Problemas comunes:

**Error: No se puede conectar a Teradata**
- Verifica credenciales en .env
- Confirma conectividad de red
- Revisa configuración del driver

**Error: MCP server no responde**
- Verifica que el servidor esté ejecutándose
- Confirma URL y puerto en configuración
- Revisa logs del MCP server

**Error: Dependencias faltantes**
- Reinstala con: `pip install -r requirements.txt`
- Verifica versión de Python (>=3.9)

**Performance lenta**
- Reduce `TOP_K_RETRIEVAL` en configuración
- Optimiza tamaño de chunks en RAG
- Considera usar GPU para embeddings

**Problemas con ChromaDB / Base de conocimiento corrupta**
- Para reiniciar completamente la base de datos vectorial ChromaDB, elimina el directorio de datos:

```bash
# Windows (PowerShell) - Recomendado
rm -r -fo data/chroma_db/

# Windows (CMD)
rmdir /s /q data\chroma_db

# Linux/macOS
rm -rf data/chroma_db/
```

- El sistema recreará automáticamente la base de datos con los documentos de `knowledge_base/` en el próximo inicio
- Esto es útil cuando:
  - Los embeddings están corruptos
  - Se han actualizado los documentos de la base de conocimiento
  - Se quiere cambiar el modelo de embeddings
  - ChromaDB presenta errores de índice o consulta

**Problemas de búsqueda inconsistente (Resuelto)**
- ✅ **Solución implementada**: Warm-up automático del modelo de embeddings
- ✅ **Auto-carga**: Los documentos se cargan automáticamente si la base está vacía
- ✅ **Validador independiente**: Usa `validate_chromadb.py` para diagnóstico completo

**Timeouts en agentes MCP (Mejorado)**
- ✅ **Circuit breaker pattern** implementado para conexiones MCP
- ✅ **Timeout protection** con fallback automático
- ✅ **Enhanced logging** para tracking detallado de timeouts

## 🏢 Solución Corporativa - Mejores Prácticas Implementadas

### 🎯 Problema Resuelto: ChromaDB en Entornos Corporativos

El proyecto implementa una **solución corporativa completa y validada** que resuelve los problemas de ChromaDB en entornos restrictivos:

#### ❌ Problemas Identificados:
- ChromaDB se colgaba en `collection.add()` en redes corporativas
- Restricciones SSL/firewall bloqueaban dependencias de red
- Problemas con `onnxruntime` y certificados corporativos
- Inicialización lenta y dependencias complejas

#### ✅ Solución Implementada:

**1. 🔧 Vector Store Corporativo Robusto**
```python
# Implementación local sin dependencias ChromaDB
- Almacenamiento persistente en archivos JSON/NumPy
- Embeddings con sentence-transformers (offline-first)  
- SSL bypass automático para certificados corporativos
- Sistema de fallbacks automáticos con 4 estrategias de carga
```

**2. ⚡ Carga Progresiva Inteligente**
```python
# Inicialización ultra-rápida en fases priorizadas
Fase 1: 10 ejemplos SQL críticos (inmediato)
Fase 2: Ejemplos SQL restantes (background)  
Fase 3: Documentación (low priority)
```

**3. 🛡️ Configuración Corporativa Optimizada**
```python
# Variables de entorno optimizadas
REQUESTS_CA_BUNDLE=""
CURL_CA_BUNDLE=""
PYTORCH_ENABLE_MPS_FALLBACK=1
OMP_NUM_THREADS=1
```

**5. 🔧 Modelo de Embeddings Robusto**
```python
# Múltiples estrategias de carga con fallbacks
- Estrategia 1: Carga directa con trust_remote_code
- Estrategia 2: Sin trust_remote_code  
- Estrategia 3: Modelo alternativo (all-MiniLM-L6-v2)
- Estrategia 4: Modelo básico de respaldo
- Warm-up automático para consistencia de resultados
```

**6. 📊 Almacenamiento Local Persistente**
```python
# Estructura de archivos optimizada
data/corporate_vector_store/
├── documents.json           # Documentos originales
├── embeddings.npy          # Arrays NumPy eficientes
├── metadata.json           # Metadatos estructurados
└── stats.json             # Estadísticas del sistema
```

### 🔧 Paquetes Adicionales para Solución Corporativa

```bash
# Dependencias específicas para entornos corporativos
sentence-transformers>=2.2.2    # Embeddings offline-first
torch>=1.13.0                   # Backend PyTorch optimizado  
numpy>=1.21.0                   # Arrays eficientes
requests>=2.28.0                # HTTP con SSL bypass
urllib3>=1.26.0                 # Conexiones robustas
```

### 🚀 Implementación en Producción

**Archivos Clave de la Solución:**
- `src/rag/vector_store.py` - **Versión productiva corporativa**
- `test/validate_chromadb.py` - **Validador independiente con auto-carga**
- `web_interface_validation.py` - **Interfaz web con validación detallada**
- `src/agents/explain_generator.py` - **Enhanced MCP con timeout protection**
- `src/agents/sql_reviewer.py` - **Revisor con timeout safety**

**Beneficios Logrados:**
1. 🚫 **SIN BLOQUEOS**: Elimina completamente los hang-ups de ChromaDB
2. 🏢 **CORPORATIVO-SAFE**: Funciona en entornos restrictivos
3. ⚡ **ARRANQUE RÁPIDO**: Inicialización inmediata
4. 💾 **PERSISTENTE**: Datos seguros entre sesiones
5. 🔧 **MANTENIBLE**: Código limpio con logging detallado
6. 📈 **ESCALABLE**: Procesamiento por lotes eficiente
7. 🛡️ **ROBUSTO**: Múltiples niveles de fallback
8. 🔄 **COMPATIBLE**: Zero breaking changes
9. 🔍 **AUTO-DIAGNÓSTICO**: Validadores independientes incluidos
10. ⚡ **BÚSQUEDA CONSISTENTE**: Warm-up automático de embeddings
11. 🎯 **TIMEOUT PROTECTION**: Circuit breakers para todas las operaciones
12. 🌐 **VALIDACIÓN DETALLADA**: Interfaz web enhanced para debugging

## 🆕 Mejoras Recientes Implementadas

### 🔧 Sistema de Validación Independiente
- **✅ Nuevo validador ChromaDB**: `test/validate_chromadb.py`
  - Auto-carga de documentos cuando la base está vacía
  - Validación completa de todas las operaciones
  - Diagnóstico detallado de performance
  - Forzado de carga si no hay documentos

### 🌐 Interfaz Web Enhanced
- **✅ Nueva interfaz detallada**: `web_interface_validation.py`
  - Salidas individuales de cada agente por separado
  - Tracking de tiempo de procesamiento individual
  - Visualización de errores y timeouts específicos
  - Validación en tiempo real de conectividad

### ⚡ Sistema RAG Optimizado
- **✅ Warm-up automático**: Inicialización del modelo de embeddings
  - Elimina el problema de "primera búsqueda sin resultados"
  - Resultados consistentes en todas las ejecuciones
  - Warm-up con múltiples queries de ejemplo
- **✅ Auto-carga inteligente**: Documentos se cargan automáticamente
- **✅ Timeout protection**: Búsquedas RAG con protección de timeout

### 🔐 MCP Enhanced con Circuit Breaker
- **✅ Timeout protection**: Protección completa contra timeouts
- **✅ Circuit breaker pattern**: Prevención automática de bloqueos
- **✅ Enhanced logging**: Tracking detallado de todas las operaciones
- **✅ Fallback modes**: Múltiples estrategias de recuperación

### 📦 Gestión de Dependencias MCP
- **✅ Consolidación completa**: Todas las dependencias MCP en `requirements.txt`
- **✅ Instalación simplificada**: Un solo comando para todas las dependencias
- **✅ Validador de requisitos**: Script de verificación automática

### 🎯 Mejoras de Performance
- **✅ Batch processing optimizado**: Tamaño de lote ajustado para mejor throughput
- **✅ Carga progresiva**: Inicialización por fases priorizadas
- **✅ Cache de embeddings**: Gestión inteligente de memoria
- **✅ SSL bypass**: Configuración optimizada para entornos corporativos

### 📊 Logging y Monitoreo
- **✅ Métricas detalladas**: Tracking completo de performance
- **✅ Error tracking**: Clasificación y seguimiento de errores
- **✅ Timeout analytics**: Análisis detallado de timeouts
- **✅ Health checks**: Validación continua de componentes

**Desarrollado con ❤️ por el equipo de Datos CRM & Filiales**