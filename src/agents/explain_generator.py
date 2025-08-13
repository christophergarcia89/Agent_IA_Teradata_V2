"""
Enhanced Explain Generator with comprehensive MCP timeout protection.
"""

import logging
import asyncio
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Importaciones necesarias
import httpx

# Importaciones MCP (opcionales)
try:
    from langchain_openai import AzureChatOpenAI
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as e:
    logging.warning(f"MCP dependencies not available: {e}")
    # Define stubs for MCP classes if not available
    ClientSession = object
    StdioServerParameters = object
    stdio_client = None

from pydantic import BaseModel, Field
from config.settings import settings
from datetime import datetime


class EnhancedExplainResult(BaseModel):
    """Enhanced result of SQL EXPLAIN operation with timeout tracking."""
    explain_plan: str = Field(description="The execution plan from EXPLAIN")
    query_cost: Optional[float] = Field(description="Estimated query cost", default=None)
    execution_time: Optional[float] = Field(description="Estimated execution time", default=None)
    warnings: List[str] = Field(description="Any warnings from the explain", default_factory=list)
    error_message: Optional[str] = Field(description="Error message if explain failed", default=None)
    success: bool = Field(description="Whether the explain was successful")
    mcp_tools_used: List[str] = Field(description="MCP tools used for the operation", default_factory=list)
    timeout_occurred: bool = Field(description="Whether any timeout occurred", default=False)
    fallback_used: bool = Field(description="Whether fallback data was used", default=False)
    connection_attempts: int = Field(description="Number of connection attempts made", default=0)
    processing_time: float = Field(description="Total processing time in seconds", default=0.0)


@dataclass
class EnhancedTeradataConnectionConfig:
    """Enhanced Teradata connection configuration with timeout settings."""
    database_uri: str
    transport_type: str = "http"  # stdio, http, sse
    server_url: Optional[str] = None
    connection_timeout: float = 15.0
    operation_timeout: float = 20.0
    max_retries: int = 3
    retry_delay: float = 2.0
    health_check_timeout: float = 5.0


class CircuitBreaker:
    """Circuit breaker pattern for MCP connections."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func):
        """Execute function through circuit breaker."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("[CIRCUIT] Attempting to reset circuit breaker")
            else:
                raise Exception(f"Circuit breaker OPEN - cooling down until {self.last_failure_time + timedelta(seconds=self.recovery_timeout)}")
        
        try:
            result = await func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == "HALF_OPEN":
            self.logger.info("[CIRCUIT] Circuit breaker reset to CLOSED")
            self.state = "CLOSED"
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"[CIRCUIT] Circuit breaker OPEN after {self.failure_count} failures")


class EnhancedTeradataMCPClient:
    """Enhanced MCP client with comprehensive timeout protection."""
    
    def __init__(self, config: EnhancedTeradataConnectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session: Optional[ClientSession] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.circuit_breaker = CircuitBreaker()
        self.connection_pool_healthy = False
        self.last_health_check = None
        self.health_check_interval = 30.0  # seconds
        
    async def safe_operation(self, operation_func, operation_name: str, timeout: float = None):
        """Safely execute MCP operation with timeout and fallback."""
        timeout = timeout or self.config.operation_timeout
        start_time = datetime.now()
        
        try:
            # Use circuit breaker
            result = await self.circuit_breaker.call(
                lambda: asyncio.wait_for(operation_func(), timeout=timeout)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"[MCP-SUCCESS] {operation_name} completed in {processing_time:.2f}s")
            
            if isinstance(result, dict):
                return result
            return {"content": str(result), "success": True}
            
        except asyncio.TimeoutError:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"[MCP-TIMEOUT] {operation_name} timed out after {timeout}s"
            self.logger.warning(error_msg)
            return {
                "error": error_msg,
                "timeout_occurred": True,
                "fallback_needed": True,
                "success": False
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"[MCP-ERROR] {operation_name} failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                "error": error_msg,
                "timeout_occurred": False,
                "fallback_needed": True,
                "success": False
            }
    
    async def initialize_with_fallback(self) -> bool:
        """Initialize MCP connection with multiple attempts and fallback."""
        max_attempts = self.config.max_retries
        
        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"[MCP-INIT] Attempt {attempt}/{max_attempts}")
                
                async def init_operation():
                    if self.config.transport_type == "stdio":
                        await self._initialize_stdio()
                    elif self.config.transport_type == "http":
                        await self._initialize_http()
                    return True
                
                result = await self.safe_operation(
                    init_operation,
                    f"MCPInitialization_Attempt{attempt}",
                    self.config.connection_timeout
                )
                
                if result.get("success", False):
                    self.logger.info(f"[MCP-INIT] Successfully initialized on attempt {attempt}")
                    return True
                    
                if attempt < max_attempts:
                    await asyncio.sleep(self.config.retry_delay * attempt)  # Exponential backoff
                    
            except Exception as e:
                self.logger.error(f"[MCP-INIT] Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(self.config.retry_delay * attempt)
        
        self.logger.warning("[MCP-INIT] All initialization attempts failed")
        return False
    
    async def _initialize_stdio(self) -> None:
        """Enhanced STDIO initialization with timeout protection."""
        # Configurar variables de entorno
        os.environ["DATABASE_URI"] = self.config.database_uri
        
        # Parámetros para el servidor Teradata MCP
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "teradata-mcp-server"],
            env={"DATABASE_URI": self.config.database_uri}
        )
        
        # Crear cliente stdio con timeout
        stdio_transport = stdio_client(server_params)
        self.session = ClientSession(stdio_transport[0], stdio_transport[1])
        
        # Initialize with timeout
        await asyncio.wait_for(
            self.session.initialize(), 
            timeout=self.config.connection_timeout
        )
        
        self.logger.info("[MCP-INIT] STDIO transport initialized successfully")
    
    async def _initialize_http(self) -> None:
        """Enhanced HTTP initialization with timeout protection."""
        if not self.config.server_url:
            raise ValueError("Server URL required for HTTP transport")
        
        # Create HTTP client with timeout
        timeout_config = httpx.Timeout(
            connect=self.config.connection_timeout / 2,
            read=self.config.operation_timeout,
            write=self.config.operation_timeout,
            pool=self.config.connection_timeout
        )
        
        self.http_client = httpx.AsyncClient(timeout=timeout_config)
        
        # Enhanced health check with retry
        for attempt in range(1, 4):  # 3 attempts for health check
            try:
                response = await asyncio.wait_for(
                    self.http_client.get(f"{self.config.server_url}/health"),
                    timeout=self.config.health_check_timeout
                )
                
                if response.status_code == 200:
                    self.connection_pool_healthy = True
                    self.last_health_check = datetime.now()
                    self.logger.info(f"[MCP-HEALTH] Health check passed on attempt {attempt}")
                    return
                else:
                    self.logger.warning(f"[MCP-HEALTH] Health check returned {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"[MCP-HEALTH] Health check attempt {attempt} failed: {e}")
                if attempt < 3:
                    await asyncio.sleep(1.0 * attempt)
        
        # Health check failed but continue - might still work for actual operations
        self.connection_pool_healthy = False
        self.logger.warning("[MCP-HEALTH] All health check attempts failed, continuing anyway")
    
    async def get_available_tools_enhanced(self) -> List[Dict[str, Any]]:
        """Get available tools with timeout protection."""
        async def get_tools_operation():
            if self.session:
                # STDIO transport
                tools = await self.session.list_tools()
                return [{"name": tool.name, "description": tool.description} for tool in tools.tools]
            elif self.http_client:
                # HTTP transport
                response = await self.http_client.get(f"{self.config.server_url}/tools")
                if response.status_code == 200:
                    return response.json().get("tools", [])
            return []
        
        result, error, timeout_occurred = await self.safe_operation(
            get_tools_operation,
            "GetAvailableTools",
            10.0  # Shorter timeout for tool listing
        )
        
        if result:
            self.logger.info(f"[MCP-TOOLS] Found {len(result)} available tools")
            return result
        else:
            self.logger.warning(f"[MCP-TOOLS] Failed to get tools: {error}")
            return []
    
    async def execute_query_enhanced(self, query: str) -> Dict[str, Any]:
        """Execute query with enhanced timeout protection and fallback."""
        async def execute_operation():
            if self.session:
                # STDIO execution
                self.logger.info(f"[MCP-EXECUTE] Executing query via STDIO: {query[:100]}...")
                result = await self.session.call_tool("execute_query", {"query": query})
                self.logger.info(f"[MCP-EXECUTE] Raw result type: {type(result)}")
                self.logger.info(f"[MCP-EXECUTE] Result content length: {len(result.content) if result.content else 0}")
                
                content = result.content[0].text if result.content else "No result"
                self.logger.info(f"[MCP-EXECUTE] Extracted content length: {len(content) if content else 0}")
                self.logger.info(f"[MCP-EXECUTE] Content preview: {content[:200]}...")
                
                return {"content": content}
            elif self.http_client:
                # HTTP execution
                self.logger.info(f"[MCP-EXECUTE] Executing query via HTTP: {query[:100]}...")
                payload = {
                    "query": query,
                    "database_uri": self.config.database_uri
                }
                response = await self.http_client.post(
                    f"{self.config.server_url}/execute",
                    json=payload
                )
                self.logger.info(f"[MCP-EXECUTE] HTTP response status: {response.status_code}")
                
                if response.status_code == 200:
                    json_result = response.json()
                    self.logger.info(f"[MCP-EXECUTE] HTTP JSON keys: {list(json_result.keys()) if json_result else 'None'}")
                    
                    # Extract the actual EXPLAIN plan content from the MCP response
                    if json_result.get('success') and json_result.get('rows'):
                        # The EXPLAIN plan is in the rows array
                        rows = json_result.get('rows', [])
                        self.logger.info(f"[MCP-EXECUTE] Processing {len(rows)} rows from EXPLAIN result")
                        
                        if rows:
                            # Convert rows to readable EXPLAIN plan format
                            plan_lines = []
                            for i, row in enumerate(rows):
                                if isinstance(row, dict):
                                    # Extract all values from the dictionary
                                    if 'explain_plan' in row:
                                        plan_lines.append(f"EXPLAIN: {row['explain_plan']}")
                                    elif 'step' in row:
                                        plan_lines.append(row['step'])
                                    elif 'metadata' in row:
                                        plan_lines.append(f"METADATA: {row['metadata']}")
                                    else:
                                        # Generic handling for other dictionary structures
                                        for key, value in row.items():
                                            plan_lines.append(f"{key.upper()}: {value}")
                                elif isinstance(row, list):
                                    plan_lines.append('\t'.join(str(cell) for cell in row))
                                else:
                                    plan_lines.append(str(row))
                            
                            explain_content = '\n'.join(plan_lines)
                            self.logger.info(f"[MCP-EXECUTE] Extracted EXPLAIN content ({len(explain_content)} chars): {explain_content[:200]}...")
                            return {"content": explain_content, "success": True, "rows": rows, "metadata": json_result.get('metadata')}
                        else:
                            self.logger.warning("[MCP-EXECUTE] No rows in successful response")
                            return {"content": "No EXPLAIN data returned", "success": True}
                    elif json_result.get('error'):
                        error_msg = json_result.get('error')
                        self.logger.error(f"[MCP-EXECUTE] MCP server error: {error_msg}")
                        return {"error": error_msg}
                    else:
                        self.logger.warning(f"[MCP-EXECUTE] Unexpected response structure: success={json_result.get('success')}, has_rows={bool(json_result.get('rows'))}")
                        # Still return the original response in case there's useful data elsewhere
                        return json_result
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    self.logger.error(f"[MCP-EXECUTE] HTTP error: {error_msg}")
                    return {"error": error_msg}
            
            self.logger.error("[MCP-EXECUTE] No transport available")
            return {"error": "No transport available"}
        
        try:
            result = await self.safe_operation(
                execute_operation,
                "ExecuteQuery",
                self.config.operation_timeout
            )
            self.logger.info(f"[MCP-EXECUTE] Final result type: {type(result)}")
            if isinstance(result, tuple):
                self.logger.info(f"[MCP-EXECUTE] Tuple result: success={result[0]}, content_preview={str(result[1])[:100] if result[1] else 'None'}")
                return result[1] if result[0] else {"error": str(result[2])}
            return result
        except Exception as e:
            error_msg = f"Execute query failed: {str(e)}"
            self.logger.error(f"[MCP-EXECUTE] Exception: {error_msg}")
            return {
                "error": error_msg,
                "timeout_occurred": False,
                "fallback_needed": True
            }
    
    async def cleanup_enhanced(self) -> None:
        """Enhanced cleanup with timeout protection."""
        cleanup_tasks = []
        
        if self.session:
            cleanup_tasks.append(self._safe_cleanup(self.session.close(), "STDIO Session"))
        
        if self.http_client:
            cleanup_tasks.append(self._safe_cleanup(self.http_client.aclose(), "HTTP Client"))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        self.logger.info("[MCP-CLEANUP] Enhanced cleanup completed")
    
    async def _safe_cleanup(self, cleanup_coro, resource_name: str):
        """Safely cleanup resource with timeout."""
        try:
            await asyncio.wait_for(cleanup_coro, timeout=5.0)
            self.logger.info(f"[MCP-CLEANUP] {resource_name} cleaned up successfully")
        except asyncio.TimeoutError:
            self.logger.warning(f"[MCP-CLEANUP] {resource_name} cleanup timed out")
        except Exception as e:
            self.logger.error(f"[MCP-CLEANUP] {resource_name} cleanup failed: {e}")


class EnhancedExplainGenerator:
    """Enhanced Explain Generator with comprehensive MCP timeout protection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client: Optional[EnhancedTeradataMCPClient] = None
        self.fallback_plans = self._load_fallback_plans()
        
        # Enhanced configuration
        self.config = EnhancedTeradataConnectionConfig(
            database_uri=getattr(settings, 'database_uri', 'teradata://localhost:1025/test'),
            transport_type=getattr(settings, 'MCP_TRANSPORT', 'http'),
            server_url=getattr(settings, 'teradata_mcp_server_url', 'http://localhost:3002'),
            connection_timeout=15.0,
            operation_timeout=20.0,
            max_retries=3,
            retry_delay=2.0,
            health_check_timeout=5.0
        )
    
    def _load_fallback_plans(self) -> Dict[str, str]:
        """Load fallback EXPLAIN plans for when MCP is unavailable."""
        return {
            "select": """
            1) Optimizer Cost: 1.5e6
            2) Operation: Full Table Scan on TABLE_NAME
            3) Estimated Rows: 10,000
            4) Join Strategy: Hash Join
            5) Index Usage: No indexes used (MCP unavailable - using fallback)
            """,
            "insert": """
            1) Optimizer Cost: 500
            2) Operation: Insert into TABLE_NAME
            3) Estimated Rows: Variable
            4) Constraints: Primary Key, Foreign Key checks
            5) Note: Actual plan unavailable (MCP timeout - using fallback)
            """,
            "update": """
            1) Optimizer Cost: 2.1e6
            2) Operation: Update TABLE_NAME
            3) Estimated Rows: Variable
            4) Index Usage: Using available indexes
            5) Note: Real-time plan unavailable (MCP connection failed - using fallback)
            """,
            "default": """
            1) Optimizer Cost: Estimated
            2) Operation: Query execution plan
            3) Note: Detailed plan unavailable due to MCP timeout
            4) Recommendation: Check MCP server connectivity
            5) Fallback: Generic execution analysis provided
            """
        }
    
    async def initialize(self) -> bool:
        """Initialize with enhanced timeout protection."""
        try:
            self.client = EnhancedTeradataMCPClient(self.config)
            success = await self.client.initialize_with_fallback()
            
            if success:
                self.logger.info("[EXPLAIN-GEN] Enhanced initialization successful")
            else:
                self.logger.warning("[EXPLAIN-GEN] MCP initialization failed, fallback mode available")
            
            return True  # Always return True because we have fallback
            
        except Exception as e:
            self.logger.error(f"[EXPLAIN-GEN] Initialization error: {e}")
            return True  # Still return True for fallback mode
    
    async def generate_explain(self, sql_query: str) -> EnhancedExplainResult:
        """Generate EXPLAIN with comprehensive timeout protection. Main entry point."""
        start_time = datetime.now()
        try:
            # Initialize result with default values
            result = EnhancedExplainResult(
                explain_plan="",
                success=False,
                error_message=None,
                query_cost=None,
                execution_time=None,
                warnings=[],
                timeout_occurred=False,
                fallback_used=False,
                connection_attempts=0,
                processing_time=0.0,
                mcp_tools_used=[]
            )

            # Try to get actual EXPLAIN from MCP
            connection_attempts = 1
            fallback_used = False
            timeout_occurred = False

            if self.client:
                explain_query = f"EXPLAIN {sql_query}"
                mcp_result = await self.client.execute_query_enhanced(explain_query)
                
                if isinstance(mcp_result, dict):
                    if mcp_result.get("error") or mcp_result.get("fallback_needed"):
                        timeout_occurred = mcp_result.get("timeout_occurred", False)
                        fallback_used = True
                        error_message = mcp_result.get("error")
                        self.logger.warning(f"[EXPLAIN] MCP failed, using fallback: {error_message}")
                    else:
                        # Success with MCP
                        result.explain_plan = str(mcp_result.get("content", "No plan available"))
                        result.success = True
                        result.mcp_tools_used = ["execute_query"]
                        result.timeout_occurred = False
                        result.fallback_used = False
                        result.connection_attempts = connection_attempts
                        result.processing_time = (datetime.now() - start_time).total_seconds()
                        return result

            # If we get here, we need to use fallback
            plan_type = self._detect_query_type(sql_query)
            fallback_plan = self.fallback_plans.get(plan_type, self.fallback_plans["default"])
            customized_plan = self._customize_fallback_plan(fallback_plan, sql_query, timeout_occurred)
            
            result.explain_plan = customized_plan
            result.success = True
            result.warnings = ["Using fallback EXPLAIN plan due to MCP unavailability"]
            result.timeout_occurred = timeout_occurred
            result.fallback_used = True
            result.connection_attempts = connection_attempts
            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"EXPLAIN generation failed: {str(e)}"
            self.logger.error(error_msg)
            
            return EnhancedExplainResult(
                explain_plan=self.fallback_plans["default"],
                success=False,
                error_message=error_msg,
                warnings=["Critical error - using minimal fallback plan"],
                timeout_occurred=timeout_occurred,
                fallback_used=True,
                connection_attempts=connection_attempts,
                processing_time=processing_time,
                query_cost=None,
                execution_time=None,
                mcp_tools_used=[]
            )

    async def generate_explain_plan(self, query: str) -> EnhancedExplainResult:
        """Generate EXPLAIN plan with comprehensive timeout protection."""
        start_time = datetime.now()
        connection_attempts = 0
        timeout_occurred = False
        fallback_used = False
        
        try:
            if self.client:
                connection_attempts = 1
                
                # Try to get actual EXPLAIN from MCP
                explain_query = f"EXPLAIN {query}"
                self.logger.info(f"[EXPLAIN-GEN] Executing EXPLAIN query: {explain_query[:100]}...")
                result = await self.client.execute_query_enhanced(explain_query)
                
                self.logger.info(f"[EXPLAIN-GEN] MCP result type: {type(result)}")
                self.logger.info(f"[EXPLAIN-GEN] MCP result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
                # Check for errors or fallback needed
                if isinstance(result, dict) and (result.get("error") or result.get("fallback_needed")):
                    timeout_occurred = result.get("timeout_occurred", False) 
                    fallback_used = True
                    error_message = result.get("error")
                    self.logger.warning(f"[EXPLAIN-GEN] MCP failed, using fallback: {error_message}")
                else:
                    # Success with MCP
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    if isinstance(result, dict):
                        # Log the full structure for debugging
                        self.logger.info(f"[EXPLAIN-GEN] Result structure: {list(result.keys()) if result else 'Empty'}")
                        
                        # Try different keys where the EXPLAIN content might be
                        explain_content = (
                            result.get("content") or 
                            result.get("result") or 
                            result.get("rows") or
                            result.get("data")
                        )
                        
                        self.logger.info(f"[EXPLAIN-GEN] Raw explain_content type: {type(explain_content)}")
                        self.logger.info(f"[EXPLAIN-GEN] Raw explain_content: {str(explain_content)[:300] if explain_content else 'None'}")
                        
                        # If explain_content is a list (rows), convert to string
                        if isinstance(explain_content, list):
                            if explain_content:  # Non-empty list
                                if isinstance(explain_content[0], dict):
                                    # Format dictionary rows nicely
                                    plan_lines = []
                                    for row in explain_content:
                                        line_parts = []
                                        for key, value in row.items():
                                            line_parts.append(f"{key}: {value}")
                                        plan_lines.append(" | ".join(line_parts))
                                    explain_content = "\n".join(plan_lines)
                                else:
                                    # Simple list, join with newlines
                                    explain_content = "\n".join(str(row) for row in explain_content)
                            else:
                                explain_content = "Empty result set"
                        
                        # Additional check for MCP success flag
                        if result.get("success") is False and result.get("error"):
                            self.logger.warning(f"[EXPLAIN-GEN] MCP reported error: {result.get('error')}")
                            fallback_used = True
                            plan_type = self._detect_query_type(query)
                            explain_content = self._customize_fallback_plan(
                                self.fallback_plans.get(plan_type, self.fallback_plans["default"]), 
                                query, 
                                False
                            )
                    else:
                        explain_content = str(result)
                    
                    self.logger.info(f"[EXPLAIN-GEN] Success! Content length: {len(str(explain_content)) if explain_content else 0}")
                    self.logger.info(f"[EXPLAIN-GEN] Content preview: {str(explain_content)[:200]}...")
                    
                    # Ensure we don't return "No plan available" if we have actual content
                    if explain_content and str(explain_content).strip() and str(explain_content).strip() not in ["No result", "Empty result set"]:
                        final_plan = str(explain_content)
                    else:
                        self.logger.warning(f"[EXPLAIN-GEN] Empty or invalid content, using fallback")
                        fallback_used = True
                        plan_type = self._detect_query_type(query)
                        final_plan = self._customize_fallback_plan(
                            self.fallback_plans.get(plan_type, self.fallback_plans["default"]), 
                            query, 
                            False
                        )
                    
                    self.logger.info(f"[EXPLAIN-GEN] Final plan being returned: {final_plan[:200]}...")
                    self.logger.info(f"[EXPLAIN-GEN] Fallback used: {fallback_used}")
                    
                    return EnhancedExplainResult(
                        explain_plan=final_plan,
                        success=True,
                        mcp_tools_used=["execute_query"],
                        timeout_occurred=False,
                        fallback_used=fallback_used,
                        connection_attempts=connection_attempts,
                        processing_time=processing_time,
                        warnings=["Using fallback plan due to empty MCP response"] if fallback_used else []
                    )
            else:
                fallback_used = True
                self.logger.info("[EXPLAIN-GEN] No MCP client available, using fallback")
            
            # Fallback plan generation
            self.logger.info("[EXPLAIN-GEN] Using fallback plan generation")
            plan_type = self._detect_query_type(query)
            fallback_plan = self.fallback_plans.get(plan_type, self.fallback_plans["default"])
            
            # Customize fallback plan with query specifics
            customized_plan = self._customize_fallback_plan(fallback_plan, query, timeout_occurred)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedExplainResult(
                explain_plan=customized_plan,
                success=True,  # Fallback is considered successful
                warnings=["Using fallback EXPLAIN plan due to MCP unavailability"],
                timeout_occurred=timeout_occurred,
                fallback_used=True,
                connection_attempts=connection_attempts,
                processing_time=processing_time,
                query_cost=None,  # No real cost available in fallback
                execution_time=None,  # No real execution time in fallback
                error_message=None,  # No error since fallback worked
                mcp_tools_used=[]  # No MCP tools used in fallback
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"EXPLAIN generation failed: {str(e)}"
            self.logger.error(f"[EXPLAIN-GEN] Exception: {error_msg}")
            
            return EnhancedExplainResult(
                explain_plan=self.fallback_plans["default"],
                success=False,
                error_message=error_msg,
                warnings=["Critical error - using minimal fallback plan"],
                timeout_occurred=timeout_occurred,
                fallback_used=True,
                connection_attempts=connection_attempts,
                processing_time=processing_time
            )
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type for appropriate fallback selection."""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            return "select"
        elif query_lower.startswith('insert'):
            return "insert" 
        elif query_lower.startswith('update'):
            return "update"
        elif query_lower.startswith('delete'):
            return "update"  # Similar pattern to update
        else:
            return "default"
    
    def _customize_fallback_plan(self, base_plan: str, query: str, timeout_occurred: bool) -> str:
        """Customize fallback plan with query-specific information."""
        customized = base_plan
        
        # Add timeout information if applicable
        if timeout_occurred:
            customized += "\n6) TIMEOUT: MCP operation timed out during execution"
        
        # Try to extract table names and add to plan
        try:
            import re
            table_matches = re.findall(r'\b(?:FROM|JOIN|UPDATE|INTO)\s+([a-zA-Z_][a-zA-Z0-9_]*)', query, re.IGNORECASE)
            if table_matches:
                customized = customized.replace("TABLE_NAME", table_matches[0])
                if len(table_matches) > 1:
                    customized += f"\n7) Additional Tables: {', '.join(table_matches[1:])}"
        except:
            pass  # If regex fails, just use original plan
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        customized += f"\n8) Generated: {timestamp} (Enhanced Fallback Mode)"
        
        return customized
    
    async def cleanup(self) -> None:
        """Enhanced cleanup."""
        if self.client:
            await self.client.cleanup_enhanced()
            self.client = None
        
        self.logger.info("[EXPLAIN-GEN] Enhanced cleanup completed")
        



# Factory function for backward compatibility
def create_explain_generator():
    """Factory function to create enhanced explain generator."""
    return EnhancedExplainGenerator()