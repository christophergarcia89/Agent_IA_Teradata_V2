"""
Enhanced Explain Generator with comprehensive MCP timeout protection.
"""

import logging
import asyncio
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

# Importaciones necesarias para MCP
try:
    from langchain_openai import AzureChatOpenAI
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import httpx
except ImportError as e:
    logging.warning(f"MCP dependencies not available: {e}")

from pydantic import BaseModel, Field
from config.settings import settings


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
            
            return result, None, False  # result, error, timeout_occurred
            
        except asyncio.TimeoutError:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"[MCP-TIMEOUT] {operation_name} timed out after {timeout}s"
            self.logger.warning(error_msg)
            return None, error_msg, True
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"[MCP-ERROR] {operation_name} failed: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg, False
    
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
                
                result, error, timeout_occurred = await self.safe_operation(
                    init_operation,
                    f"MCPInitialization_Attempt{attempt}",
                    self.config.connection_timeout
                )
                
                if result:
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
                result = await self.session.call_tool("execute_query", {"query": query})
                return {"content": result.content[0].text if result.content else "No result"}
            elif self.http_client:
                # HTTP execution
                payload = {
                    "query": query,
                    "database_uri": self.config.database_uri
                }
                response = await self.http_client.post(
                    f"{self.config.server_url}/execute",
                    json=payload
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"HTTP {response.status_code}: {response.text}"}
            
            return {"error": "No transport available"}
        
        result, error, timeout_occurred = await self.safe_operation(
            execute_operation,
            "ExecuteQuery",
            self.config.operation_timeout
        )
        
        if result:
            return result
        else:
            # Return error information for fallback handling
            return {
                "error": error,
                "timeout_occurred": timeout_occurred,
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
            database_uri=getattr(settings, 'DATABASE_URI', 'teradata://localhost:1025/test'),
            transport_type=getattr(settings, 'MCP_TRANSPORT', 'http'),
            server_url=getattr(settings, 'MCP_SERVER_URL', 'http://localhost:8000'),
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
                result = await self.client.execute_query_enhanced(explain_query)
                
                if result.get("error") or result.get("fallback_needed"):
                    timeout_occurred = result.get("timeout_occurred", False)
                    fallback_used = True
                    self.logger.warning(f"[EXPLAIN] MCP failed, using fallback: {result.get('error')}")
                else:
                    # Success with MCP
                    processing_time = (datetime.now() - start_time).total_seconds()
                    return EnhancedExplainResult(
                        explain_plan=str(result.get("content", "No plan available")),
                        success=True,
                        mcp_tools_used=["execute_query"],
                        timeout_occurred=False,
                        fallback_used=False,
                        connection_attempts=connection_attempts,
                        processing_time=processing_time
                    )
            else:
                fallback_used = True
                self.logger.info("[EXPLAIN] No MCP client available, using fallback")
            
            # Fallback plan generation
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
                fallback_used=fallback_used,
                connection_attempts=connection_attempts,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"EXPLAIN generation failed: {str(e)}"
            
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
