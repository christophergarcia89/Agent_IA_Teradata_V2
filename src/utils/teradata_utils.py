"""
Teradata database utilities
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from contextlib import asynccontextmanager
import asyncio

try:
    import teradatasql
except ImportError:
    teradatasql = None

from config.settings import settings


class TeradataConnection:
    """Teradata database connection manager."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection = None
        
    async def connect(self) -> None:
        """Establish connection to Teradata."""
        if teradatasql is None:
            raise ImportError("teradatasql package is required for Teradata connectivity")
        
        try:
            connection_params = {
                'host': settings.teradata_host,
                'user': settings.teradata_user,
                'password': settings.teradata_password,
                'database': settings.teradata_database,
                'logdata': 'teradata.log' if settings.debug else None
            }
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.connection = await loop.run_in_executor(
                None, 
                lambda: teradatasql.connect(**connection_params)
            )
            
            self.logger.info("Connected to Teradata successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Teradata: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Teradata connection."""
        if self.connection:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.connection.close)
                self.logger.info("Disconnected from Teradata")
            except Exception as e:
                self.logger.error(f"Error disconnecting from Teradata: {e}")
            finally:
                self.connection = None
    
    async def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute a query and return results with column names."""
        if not self.connection:
            raise ValueError("Not connected to Teradata")
        
        try:
            loop = asyncio.get_event_loop()
            
            # Execute query in thread pool
            cursor = await loop.run_in_executor(None, self.connection.cursor)
            await loop.run_in_executor(None, cursor.execute, query)
            
            # Fetch results
            rows = await loop.run_in_executor(None, cursor.fetchall)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Convert to dictionaries
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            await loop.run_in_executor(None, cursor.close)
            
            self.logger.debug(f"Query executed successfully, returned {len(results)} rows")
            return results, columns
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
    
    async def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate query syntax without executing."""
        try:
            # Use EXPLAIN to validate syntax
            explain_query = f"EXPLAIN {query}"
            await self.execute_query(explain_query)
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self.connection:
            raise ValueError("Not connected to Teradata")
        
        try:
            # Start transaction
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.connection.execute, "BEGIN TRANSACTION")
            
            yield self
            
            # Commit transaction
            await loop.run_in_executor(None, self.connection.execute, "COMMIT")
            
        except Exception as e:
            # Rollback on error
            try:
                await loop.run_in_executor(None, self.connection.execute, "ROLLBACK")
            except:
                pass
            raise e


class QueryValidator:
    """Utility class for SQL query validation."""
    
    @staticmethod
    def is_select_query(query: str) -> bool:
        """Check if query is a SELECT statement."""
        return query.strip().upper().startswith('SELECT')
    
    @staticmethod
    def is_explain_query(query: str) -> bool:
        """Check if query is an EXPLAIN statement."""
        return query.strip().upper().startswith('EXPLAIN')
    
    @staticmethod
    def extract_table_names(query: str) -> List[str]:
        """Extract table names from SQL query."""
        import re
        
        # Simple regex to find table names (this is basic and may need improvement)
        # Matches FROM and JOIN clauses
        pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        
        return list(set(matches))  # Remove duplicates
    
    @staticmethod
    def estimate_complexity(query: str) -> str:
        """Estimate query complexity based on content."""
        query_upper = query.upper()
        
        # Count complexity indicators
        joins = query_upper.count('JOIN')
        subqueries = query_upper.count('SELECT') - 1  # Subtract main SELECT
        unions = query_upper.count('UNION')
        window_functions = query_upper.count('OVER(')
        
        complexity_score = joins + (subqueries * 2) + (unions * 1.5) + (window_functions * 1.5)
        
        if complexity_score == 0:
            return "Simple"
        elif complexity_score <= 3:
            return "Moderate"
        elif complexity_score <= 7:
            return "Complex"
        else:
            return "Very Complex"


class TeradataOptimizer:
    """Utility class for Teradata-specific optimizations."""
    
    @staticmethod
    def suggest_primary_index(table_columns: List[str], query_patterns: List[str]) -> List[str]:
        """Suggest primary index columns based on query patterns."""
        suggestions = []
        
        # This is a simplified version - in practice, this would be more sophisticated
        for pattern in query_patterns:
            if 'WHERE' in pattern.upper():
                # Extract WHERE conditions and suggest PI based on most frequent equality conditions
                # This is a placeholder for more complex logic
                suggestions.append("Consider primary index on frequently filtered columns")
        
        return suggestions
    
    @staticmethod
    def suggest_statistics(query: str) -> List[str]:
        """Suggest statistics to collect based on query patterns."""
        suggestions = []
        
        query_upper = query.upper()
        
        if 'JOIN' in query_upper:
            suggestions.append("Collect statistics on join columns")
        
        if 'WHERE' in query_upper:
            suggestions.append("Collect statistics on WHERE clause columns")
        
        if 'GROUP BY' in query_upper:
            suggestions.append("Collect statistics on GROUP BY columns")
        
        if 'ORDER BY' in query_upper:
            suggestions.append("Consider secondary indexes for ORDER BY columns")
        
        return suggestions
    
    @staticmethod
    def detect_anti_patterns(query: str) -> List[str]:
        """Detect common Teradata anti-patterns."""
        anti_patterns = []
        query_upper = query.upper()
        
        # Check for product joins
        if 'CROSS JOIN' in query_upper or ('FROM' in query_upper and ',' in query_upper and 'WHERE' not in query_upper):
            anti_patterns.append("Potential product join detected")
        
        # Check for functions on indexed columns
        if any(func in query_upper for func in ['SUBSTR(', 'UPPER(', 'LOWER(', 'TRIM(']):
            anti_patterns.append("Functions on columns may prevent index usage")
        
        # Check for SELECT *
        if 'SELECT *' in query_upper:
            anti_patterns.append("SELECT * may retrieve unnecessary columns")
        
        # Check for NOT IN with potential NULLs
        if 'NOT IN' in query_upper:
            anti_patterns.append("NOT IN may behave unexpectedly with NULL values")
        
        return anti_patterns
