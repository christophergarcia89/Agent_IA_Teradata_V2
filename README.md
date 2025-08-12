# 🔍 Teradata SQL Agent

Sistema inteligente multi-agente de análisis y optimización de queries SQL para Teradata basado en LangGraph, LCEL y técnicas RAG con conectividad real validada.

## 📋 Descripción

Este proyecto implementa un sistema multi-agente productivo que utiliza LangGraph para orquestar tres agentes especializados con conectividad real a Teradata:

1. **🔍 SQL Reviewer Agent**: Revisa queries usando RAG con base de conocimiento interna de estándares
2. **📊 Explain Generator Agent**: Genera planes EXPLAIN usando MCP Server con conexión real a Teradata
3. **🎯 Explain Interpreter Agent**: Interpreta planes y proporciona sugerencias de optimización específicas

## 🏗️ Arquitectura

```
┌─────────────────┐    ┌──────────────────┐     ┌─────────────────────┐
│  SQL Reviewer   │───▶│ Explain Generator│───▶│ Explain Interpreter │
│    (RAG)        │    │   (MCP Server)   │     │   (Optimization)    │
│  ChromaDB       │    │    Teradata      │     │  Azure OpenAI       │
└─────────────────┘    └──────────────────┘     └─────────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                              │
│                     (LCEL Orchestration)                          │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Teradata Database (IP: 10.33.84.36)                   │
│         Conectividad Validada - Versión 17.20.03.28                │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Características

- ✅ **Conectividad Real Validada** con Teradata Database (IP: 10.33.84.36)
- 🔍 **Revisión de estándares** usando base de conocimiento interna con ChromaDB
- 📊 **Generación de planes EXPLAIN** con MCP Server y teradatasql library
- 🎯 **Análisis inteligente** de performance con sugerencias específicas
- 🔄 **Workflow automatizado** con LangGraph y circuit breaker patterns
- 🌐 **Interfaz web enhanced** con protección de timeout y fallback
- 📝 **Logging completo** con métricas detalladas de performance
- 🏗️ **Arquitectura modular** con manejo robusto de errores
- 🔐 **Timeout Protection** y circuit breaker para conexiones MCP
- 🎛️ **Modo Fallback** para operaciones sin MCP Server

## 📁 Estructura del Proyecto

```
Agent_IA_Teradata/
├── src/
│   ├── agents/
│   │   ├── sql_reviewer.py              # Agente revisor con RAG
│   │   ├── explain_generator.py         # Agente generador EXPLAIN (✅ Validado)
│   │   ├── explain_generator_enhanced.py # Versión con circuit breaker
│   │   ├── explain_interpreter.py       # Agente intérprete
│   │   └── workflow.py                  # Orquestación LangGraph
│   ├── rag/
│   │   ├── document_loader.py           # Carga de documentos
│   │   └── vector_store.py              # ChromaDB vectorial
│   ├── utils/
│   │   ├── teradata_utils.py            # Utilidades Teradata
│   │   ├── enhanced_azure_openai_utils.py # Utils Azure OpenAI
│   │   └── logging_utils.py             # Sistema de logging
│   ├── web_interface.py                 # Interfaz web básica
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
│   ├── test_direct_teradata_basic.py    # ✅ Test conectividad validado
│   ├── test_mcp_validation.py           # ✅ Test validación MCP
│   └── test_azure_connection.py         # Test Azure OpenAI
├── logs/                                # Directorio de logs
│   ├── teradata_agent.log
│   ├── sql_queries.log
│   ├── performance.log
│   └── errors.log
├── mcp_server.py                        # ✅ MCP Server real validado
├── main.py                              # Punto de entrada principal
├── requirements.txt                     # Dependencias Python
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
cd Agent_IA_Teradata

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Copia el contenido a `.env`:

```properties
# Azure OpenAI API Configuration (✅ Validado)
AZURE_OPENAI_API_KEY=tu_clave_azure_openai
AZURE_OPENAI_ENDPOINT=https://tu-endpoint.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Teradata Database Configuration (✅ Validado)
TERADATA_HOST=EDW
TERADATA_USER=Usr_Mkt_Common
TERADATA_PASSWORD=DR2012td
TERADATA_DATABASE=teraprod.bci.cl

# Teradata MCP Server Configuration (✅ Validado)
DATABASE_URI=teradata://Usr_Mkt_Common:DR2012td@10.33.84.36:1025/EDW?sslmode=disable&cop=off
TERADATA_MCP_SERVER_URL=http://localhost:3002
MCP_TRANSPORT_TYPE=http

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# LangGraph Configuration
MAX_ITERATIONS=10
TIMEOUT_SECONDS=300
```

### 4. Verificar conectividad

```bash
# Test de conexión directa (✅ Validado)
python test_direct_teradata_basic.py

# Test de validación MCP (✅ Validado)
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

Abre tu navegador en: `http://localhost:8002`

### 3. Aplicación Principal

```bash
# Terminal 3: Aplicación principal
python main.py
```

### 4. Interfaz Web Básica

```bash
python src/web_interface.py
```

### 5. CLI de Revisor SQL

```bash
python sql_reviewer_cli.py
```

## 🧪 Testing y Validación

### Tests de Conectividad Validados

```bash
# ✅ Test conexión directa Teradata (VALIDADO)
python test_direct_teradata_basic.py

# Resultado esperado:
# ✅ CONEXIÓN EXITOSA en X.XX segundos
# Usuario: USR_MKT_COMMON
# Database: USR_MKT_COMMON
# Session: XXXXXXXX
# Versión: 17.20.03.28
```

```bash
# ✅ Test validación MCP completa (VALIDADO)
python test\test_mcp_validation.py

# Resultado esperado:
# ✅ EXPLAIN exitoso en X.XX segundos
# 🔧 Tools MCP usadas: 1
# 📄 Plan preview: -> The row is sent directly back...
# 🎉 RESULTADO: ✅ MCP VALIDADO EXITOSAMENTE
```

### Tests Adicionales

```bash
# Test Azure OpenAI
python test\test_azure_connection.py

# Test instalación dependencias MCP
python install_mcp_dependencies.py
```

## 🔧 Configuración Técnica Validada

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

## 📊 Ejemplos de Uso Validados

### Ejemplo 1: EXPLAIN Query Simple


TERADATA_PASSWORD=tu_password
TERADATA_DATABASE=tu_database

# MCP Server
TERADATA_MCP_SERVER_URL=http://localhost:3000
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
curl -X POST "http://localhost:8000/analyze" \\
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

## 🧪 Testing

### Ejecutar tests unitarios

```bash
pytest tests/
```

### Test de integración

```bash
python -m pytest tests/test_integration.py -v
```

### Test de performance

```bash
python tests/benchmark.py
```

## 🔌 Integración con MCP Server

Para usar el teradata-mcp-server:

1. Instala y configura teradata-mcp-server
2. Inicia el servidor en puerto 3000
3. Configura `TERADATA_MCP_SERVER_URL` en .env

Ejemplo de configuración MCP:

```json
{
  "host": "localhost",
  "port": 3000,
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

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Guías de contribución:

- Sigue las convenciones de código existentes
- Agrega tests para nuevas funcionalidades
- Actualiza documentación cuando sea necesario
- Usa mensajes de commit descriptivos

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **LangChain/LangGraph** - Framework de orquestación
- **OpenAI** - Modelos de lenguaje
- **Teradata** - Motor de base de datos
- **ChromaDB** - Base de datos vectorial
- **FastAPI** - Framework web

## 📞 Soporte

Para soporte y preguntas:

- 📧 Email: equipo-desarrollo@empresa.com
- 💬 Slack: #teradata-sql-agent
- 📝 Issues: GitHub Issues
- 📖 Wiki: Documentación interna

---

**Desarrollado con ❤️ por el equipo de Data Engineering**
