"""
Agente Revisor de Consultas SQL usando RAG para cumplimiento de estándares.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.rag.vector_store import VectorStore
from src.utils.azure_openai_utils import azure_openai_manager
from config.settings import settings


class SQLReviewResult(BaseModel):
    """Result of SQL query review."""
    is_compliant: bool = Field(description="Whether the query follows standards")
    violations: List[str] = Field(description="List of standard violations found")
    corrected_query: str = Field(description="Corrected version of the query")
    recommendations: List[str] = Field(description="Additional recommendations")
    confidence_score: float = Field(description="Confidence in the review (0-1)")
    used_examples: List[str] = Field(description="Examples used for reference")
    rag_mode: str = Field(description="RAG mode used: 'full' or 'partial'")


@dataclass
class ReviewContext:
    """Context for SQL review."""
    original_query: str
    similar_examples: List[Document]
    documentation: List[Document]
    ok_examples: List[Document]


class SQLReviewerAgent:
    """Agent that reviews SQL queries for standards compliance using RAG."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None, use_full_rag: bool = False):
        self.logger = logging.getLogger(__name__)
        self.llm = azure_openai_manager.get_llm()
        self.use_full_rag = use_full_rag
        
        # Use provided vector store or create a new one
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore()
            
        self.output_parser = PydanticOutputParser(pydantic_object=SQLReviewResult)
        
        # Create the review prompt
        self.review_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt())
        ])
        
    async def initialize(self) -> None:
        """Initialize the agent with timeout protection."""
        try:
            if self.vector_store:
                # Try to initialize vector store with timeout
                import asyncio
                try:
                    await asyncio.wait_for(
                        self.vector_store.initialize(),
                        timeout=30.0
                    )
                    self.logger.info("SQL Reviewer Agent initialized with vector store")
                except asyncio.TimeoutError:
                    self.logger.warning("Vector store initialization timed out - agent will work without full RAG")
                except Exception as vs_error:
                    self.logger.warning(f"Vector store initialization failed: {vs_error} - agent will work without full RAG")
            else:
                self.logger.info("SQL Reviewer Agent initialized without vector store")
                
        except Exception as e:
            self.logger.error(f"Error initializing SQL Reviewer Agent: {e}")
            # Don't raise - agent can work without vector store
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for SQL review."""
        return """Eres un experto en ingeniería de datos especializado en estandarización de código SQL para Data Engineers. Tu función es ayudar a los usuarios a ajustar su código conforme a las mejores prácticas establecidas, utilizando EXCLUSIVAMENTE la base de conocimiento proporcionada en los datos adjuntos.

## Base de conocimiento

La base de conocimiento interno contiene documentación de mejores prácticas y ejemplos etiquetados como:

- **OK**: Representan el estándar correcto a seguir
- **NOK**: Representan código que viola los estándares y debe ser corregido

Cada ejemplo NOK tiene su correspondiente versión OK (ejemplo: "UPDATE - Ejemplo 1 NOK.sql" se corrige siguiendo "UPDATE - Ejemplo 1 OK.sql", "UPDATE - Ejemplo 2 A NOK.sql" / "UPDATE - Ejemplo 2 B NOK.sql" / "UPDATE - Ejemplo 2 C NOK.sql" se corrige siguiendo "UPDATE - Ejemplo 2 OK.sql", "CREATE - Ejemplo 1 NOK.sql" se corrige siguiendo "CREATE - Ejemplo 1 OK.sql").

## Restricciones críticas

1. **PROHIBIDO el uso de sentencias UPDATE**:

- Impacto negativo en rendimiento
- Para modificar registros, utiliza el siguiente método estandarizado:
    a) Borrar tabla temporal si existe
    b) Crear tabla temporal con estructura necesaria
    c) Realizar INSERT en tabla temporal con información resultante de una consulta

2. **PROHIBIDO el uso de los siguientes esquemas**:

- BCIMKT
- MKT_DESA_JOURNEYBUILDER_TB
- MKT_CRM_CMP_TB
- DOM_BCI_DSR_TB
- DOM_BCI_DSR_VW
- MKT_EXPLORER_TB
- EDW_TEMPUSU
- ARMVIEWS

## Validaciones por tipo de sentencia

### SELECT - Reglas de estandarización:

- Keywords en MAYÚSCULAS y sin espacios extra
- No usar punto y coma final
- Indentación correcta (4 espacios)
- Prohibido usar asterisco (*) en SELECT
- Comas a la IZQUIERDA en columnas
- JOIN con ON en línea separada
- AND/OR/NOT deben ir a la IZQUIERDA
- Alias de tabla exactamente 3 caracteres
- GROUP BY usando nombres de campos, no posiciones numéricas
- CASE WHEN en formato vertical para múltiples condiciones
- Mantener sangría después del WHERE
- Evitar transformación de datos en WHERE
- Reemplazar "<>" por "NOT(campo1=campo2)"
- No usar LIKE (sustituir por SUBSTR, POSITION)
- No usar IN
- No usar OR (sustituir por UNION ALL)

### CREATE - Reglas de estandarización:

- No permitir creación implícita mediante CREATE AS SELECT
- Tablas temporales solo con esquemas temporales
- Tablas temporales "T" deben eliminarse al final de BTEQ
- Utilizar MULTISET

### Reglas transversales:

- Validar resultado después de cada ejecución: .IF ERRORCODE <> 0 THEN .QUIT n;
- Borrar tablas temporales al final de BTEQ
- Formato RUT: DECIMAL(8,0)
- Formato DATE: DATE FORMAT 'YYYY-MM-DD'
- Formato campos texto: VARCHAR(n) CHARACTER SET LATIN NOT CASESPECIFIC
- Reemplazar FLOAT por DECIMAL
- No usar CURRENT_DATE
- Prohibido código comentado
- Prohibido usar DISTINCT (reemplazar por GROUP BY)

FORMATO DE SALIDA:
{format_instructions}

Sé específico en tus observaciones y proporciona explicaciones claras de por qué algo viola los estándares."""

    def _get_human_prompt(self) -> str:
        """Get the human prompt template."""
        return """Analiza el siguiente query SQL de Teradata:

QUERY A REVISAR:
```sql
{original_query}
```

EJEMPLOS SIMILARES DE LA BASE DE CONOCIMIENTO:
{similar_examples}

DOCUMENTACIÓN RELEVANTE:
{documentation}

EJEMPLOS DE BUENAS PRÁCTICAS (OK):
{ok_examples}

## Metodología de respuesta

1. Al recibir código SQL, identifica el tipo de sentencia (SELECT, CREATE, etc.)
2. Busca en el sistema interno ejemplos similares marcados como "OK" como referencia para corregir y "NOK" para seguir a la corrección del código
3. Identifica las violaciones de estándares presentes en el código
4. Proporciona la versión corregida, siguiendo los ejemplos "OK" como referencia. De igual manera incluye la corrección con los esquemas prohibidos, pero indicando que deben ser evitados.
5. Explica brevemente las correcciones realizadas

Si la consulta no está relacionada con estándares de código SQL, indicar que sólo puedes ayudar con estandarización de código SQL.

Si no encuentras ejemplos relevantes en la base de conocimiento, responde que no tienes suficiente contexto para ese caso específico, sin inventar respuestas."""

    async def review_query(self, sql_query: str) -> SQLReviewResult:
        """Review a SQL query for standards compliance - REQUIRES fully loaded vector store."""
        try:
            self.logger.info("Starting SQL query review")
            
            # STRICT REQUIREMENT: Vector store must be ready and loaded
            if not self.vector_store or not hasattr(self.vector_store, '_corporate_store') or not self.vector_store._corporate_store:
                raise RuntimeError("Vector store not initialized - RAG system required for SQL review")
            
            # Verify vector store has documents
            doc_count = await self.vector_store._get_collection_count()
            if doc_count == 0:
                raise RuntimeError(f"Vector store is empty ({doc_count} documents) - cannot perform standards review without knowledge base")
            
            self.logger.info(f"[SUCCESS] Vector store ready with {doc_count} documents")
            
            # Gather context from RAG - this is now guaranteed to work
            context = await self._gather_review_context_robust(sql_query)
            
            # Format the prompt
            formatted_prompt = self.review_prompt.format_messages(
                original_query=sql_query,
                similar_examples=self._format_examples(context.similar_examples),
                documentation=self._format_documentation(context.documentation),
                ok_examples=self._format_examples(context.ok_examples),
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # Get LLM response with timeout
            import asyncio
            response = await asyncio.wait_for(
                self.llm.ainvoke(formatted_prompt),
                timeout=30.0  # 30 second timeout for LLM
            )
            
            # Parse the result
            result = self.output_parser.parse(response.content)
            
            # Add RAG mode to result 
            result_dict = result.model_dump()
            result_dict['rag_mode'] = 'full' if self.use_full_rag else 'partial'
            
            # Create new result with RAG mode included
            final_result = SQLReviewResult(**result_dict)
            
            self.logger.info(f"SQL review completed. Compliant: {final_result.is_compliant}")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error reviewing SQL query: {e}")
            # Return a default result on error
            return SQLReviewResult(
                is_compliant=False,
                violations=[f"Error during review: {str(e)}"],
                corrected_query=sql_query,
                recommendations=["Manual review required due to processing error"],
                confidence_score=0.0,
                used_examples=[],
                rag_mode="none"  # Error case, no RAG was used
            )
    
    async def _ensure_vector_store_ready(self):
        """Ensure vector store is properly initialized and ready."""
        try:
            if not self.vector_store:
                self.logger.error("No vector store configured")
                return
                
            # Check if vector store is initialized
            if not hasattr(self.vector_store, '_vector_store') or self.vector_store._vector_store is None:
                self.logger.info("Vector store not ready, initializing...")
                await self.vector_store.initialize()
                # Give it time to complete background loading
                import asyncio
                await asyncio.sleep(3)
                self.logger.info("Vector store initialization completed")
            else:
                self.logger.debug("Vector store already initialized")
        except Exception as e:
            self.logger.warning(f"Vector store initialization issue: {e}")
            # Continue without vector store if needed
    
    async def _gather_review_context_robust(self, sql_query: str) -> ReviewContext:
        """Gather context with GUARANTEED RAG availability - no fallbacks."""
        # Vector store is guaranteed to be ready at this point
        query_type = self._detect_query_type(sql_query)
        
        try:
            # Get similar examples - REQUIRED
            similar_examples = await self.vector_store.search_similar_examples(sql_query, k=5)
            
            # Get documentation - REQUIRED  
            documentation = await self.vector_store.search_documentation(f"{query_type} standards", k=3)
            
            # Get OK examples - REQUIRED
            ok_examples = await self.vector_store.search_ok_examples_by_category(
                f"{query_type} SQL examples", category=query_type, k=5
            )
            
            context = ReviewContext(
                original_query=sql_query,
                similar_examples=similar_examples,
                documentation=documentation,
                ok_examples=ok_examples
            )
            
            # Log context quality
            total_context_items = len(similar_examples) + len(documentation) + len(ok_examples)
            self.logger.info(f"[SUCCESS] Context gathered: {total_context_items} items (similar: {len(similar_examples)}, docs: {len(documentation)}, OK: {len(ok_examples)})")
            
            # Validate we have enough context for quality review
            if total_context_items < 3:
                self.logger.warning(f"[WARNING] Limited context available ({total_context_items} items) - review quality may be reduced")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error gathering robust context: {e}")
            # Since we require RAG, re-raise the exception
            raise RuntimeError(f"Failed to gather RAG context: {e}")

    async def _gather_review_context_safe(self, sql_query: str) -> ReviewContext:
        """Gather context with timeout protection and fallback options."""
        # Always create a basic context first
        context = ReviewContext(
            original_query=sql_query,
            similar_examples=[],
            documentation=[],
            ok_examples=[]
        )
        
        # Quick vector store readiness check
        if not self.vector_store:
            self.logger.warning("No vector store available - using basic review")
            return context
        
        # Detect query type for potential fallback
        query_type = self._detect_query_type(sql_query)
        
        try:
            # Ultra-fast vector store queries with very short timeouts
            import asyncio
            
            # Try to get similar examples with short timeout
            try:
                similar_examples = await asyncio.wait_for(
                    self.vector_store.search_similar_examples(sql_query, k=3),
                    timeout=5.0  # 5 seconds max
                )
                context.similar_examples = similar_examples
                self.logger.debug(f"Found {len(similar_examples)} similar examples")
                
            except asyncio.TimeoutError:
                self.logger.info("Similar examples search timed out - continuing without them")
            except Exception as ex_error:
                self.logger.warning(f"Similar examples search failed: {ex_error}")
            
            # Try to get documentation with short timeout
            try:
                documentation = await asyncio.wait_for(
                    self.vector_store.search_documentation(f"{query_type} standards", k=2),
                    timeout=3.0  # 3 seconds max
                )
                context.documentation = documentation
                self.logger.debug(f"Found {len(documentation)} documentation pieces")
                
            except asyncio.TimeoutError:
                self.logger.info("Documentation search timed out - continuing without it")
            except Exception as doc_error:
                self.logger.warning(f"Documentation search failed: {doc_error}")
            
            # Try to get OK examples with short timeout
            try:
                ok_examples = await asyncio.wait_for(
                    self.vector_store.search_ok_examples_by_category(
                        f"{query_type} SQL examples", category=query_type, k=3
                    ),
                    timeout=5.0  # 5 seconds max
                )
                context.ok_examples = ok_examples
                self.logger.debug(f"Found {len(ok_examples)} OK examples")
                
            except asyncio.TimeoutError:
                self.logger.info("OK examples search timed out - continuing without them")
            except Exception as ok_error:
                self.logger.warning(f"OK examples search failed: {ok_error}")
        
        except Exception as e:
            self.logger.warning(f"Error in context gathering: {e} - using basic context")
        
        # Log context summary
        total_context_items = (
            len(context.similar_examples) + 
            len(context.documentation) + 
            len(context.ok_examples)
        )
        self.logger.info(f"Context gathered: {total_context_items} items total")
        
        return context

    async def _gather_review_context(self, sql_query: str) -> ReviewContext:
        """Gather relevant context for SQL review."""
        # Check if vector store is provided
        if not self.vector_store:
            self.logger.error("Vector store not provided to SQL Reviewer")
            return ReviewContext(
                original_query=sql_query,
                similar_examples=[],
                documentation=[],
                ok_examples=[]
            )
        
        # Detect query type (UPDATE, SELECT, CREATE, etc.)
        query_type = self._detect_query_type(sql_query)
        
        try:
            # Search for similar examples
            similar_examples = await self.vector_store.search_similar_examples(
                sql_query, k=settings.top_k_retrieval
            )
            
            # Search for relevant documentation
            documentation = await self.vector_store.search_documentation(
                f"{query_type} SQL best practices Teradata", k=3
            )
            
            # Get OK examples for the detected category
            ok_examples = await self.vector_store.search_ok_examples_by_category(
                f"{query_type} SQL examples", category=query_type, k=5
            )
            
        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            # Return empty results if search fails
            similar_examples = []
            documentation = []
            ok_examples = []
        
        return ReviewContext(
            original_query=sql_query,
            similar_examples=similar_examples,
            documentation=documentation,
            ok_examples=ok_examples
        )
    
    def _detect_query_type(self, sql_query: str) -> str:
        """Detect the type of SQL query."""
        query_upper = sql_query.upper().strip()
        
        if query_upper.startswith('SELECT'):
            return 'SELECT'
        elif query_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif query_upper.startswith('INSERT'):
            return 'INSERT'
        elif query_upper.startswith('DELETE'):
            return 'DELETE'
        elif query_upper.startswith('CREATE'):
            return 'CREATE'
        elif query_upper.startswith('ALTER'):
            return 'ALTER'
        elif query_upper.startswith('DROP'):
            return 'DROP'
        else:
            return 'UNKNOWN'
    
    def _format_examples(self, examples: List[Document]) -> str:
        """Format examples for prompt inclusion."""
        if not examples:
            return "No hay ejemplos disponibles."
        
        formatted = []
        for i, example in enumerate(examples[:5], 1):  # Limit to 5 examples
            metadata = example.metadata
            example_type = metadata.get('example_type', 'Unknown')
            category = metadata.get('category', 'Unknown')
            filename = metadata.get('filename', 'Unknown')
            
            formatted.append(f"""
Ejemplo {i} ({example_type} - {category}):
Archivo: {filename}
```sql
{example.page_content}
```
""")
        
        return "\n".join(formatted)
    
    def _format_documentation(self, docs: List[Document]) -> str:
        """Format documentation for prompt inclusion."""
        if not docs:
            return "No hay documentación disponible."
        
        formatted = []
        for i, doc in enumerate(docs[:3], 1):  # Limit to 3 docs
            filename = doc.metadata.get('filename', 'Unknown')
            formatted.append(f"""
Documento {i}: {filename}
{doc.page_content}
""")
        
        return "\n".join(formatted)
    
    async def get_correction_suggestions(self, sql_query: str) -> Dict[str, Any]:
        """Get specific correction suggestions for a query."""
        try:
            # Get related OK/NOK pairs
            pairs = await self.vector_store.get_related_pairs(sql_query)
            
            suggestions = {}
            for pair_key, pair_docs in pairs.items():
                ok_docs = pair_docs.get("OK", [])
                nok_docs = pair_docs.get("NOK", [])
                
                if ok_docs and nok_docs:
                    suggestions[pair_key] = {
                        "correct_example": ok_docs[0].page_content,
                        "incorrect_example": nok_docs[0].page_content,
                        "category": ok_docs[0].metadata.get("category", "Unknown")
                    }
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting correction suggestions: {e}")
            return {}

    async def analyze_query(self, sql_query: str) -> SQLReviewResult:
        """Alias for review_query for compatibility."""
        return await self.review_query(sql_query)
