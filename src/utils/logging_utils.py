"""
Logging utilities for the Teradata SQL Agent project.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'query_id'):
            log_entry["query_id"] = record.query_id
        if hasattr(record, 'performance_metrics'):
            log_entry["performance_metrics"] = record.performance_metrics
        
        return json.dumps(log_entry, ensure_ascii=False)


class QueryLogFilter(logging.Filter):
    """Filter for SQL query-related logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter records related to SQL queries."""
        return hasattr(record, 'query_type') or 'sql' in record.getMessage().lower()


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json_logging: bool = False,
    enable_performance_logging: bool = True
) -> None:
    """Setup comprehensive logging for the application."""
    
    # Get log level from settings or parameter
    level = log_level or settings.log_level
    log_level_enum = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_enum)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_enum)
    
    # Ensure UTF-8 encoding for console output
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Fallback if reconfigure is not available
    
    if enable_json_logging:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general logs with UTF-8 encoding
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / (log_file or "teradata_agent.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Ensure UTF-8 encoding for file output
    )
    file_handler.setLevel(log_level_enum)
    file_handler.setFormatter(console_formatter)
    root_logger.addHandler(file_handler)
    
    # Separate handler for SQL queries with UTF-8 encoding
    query_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "sql_queries.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'  # Ensure UTF-8 encoding
    )
    query_handler.setLevel(logging.INFO)
    query_handler.addFilter(QueryLogFilter())
    query_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(query_handler)
    
    # Performance logging
    if enable_performance_logging:
        performance_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "performance.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding='utf-8'  # Ensure UTF-8 encoding
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(JSONFormatter())
        
        # Add to performance logger specifically
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(performance_handler)
        perf_logger.setLevel(logging.INFO)
    
    # Error logging
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "errors.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8'  # Ensure UTF-8 encoding
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)
    
    # Set specific log levels for external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    logging.info("Logging setup completed")


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
    
    def log_query_performance(
        self,
        query_id: str,
        query_type: str,
        execution_time: float,
        query_length: int,
        agent_used: str,
        success: bool,
        additional_metrics: Optional[dict] = None
    ) -> None:
        """Log query performance metrics."""
        
        metrics = {
            "query_id": query_id,
            "query_type": query_type,
            "execution_time_seconds": execution_time,
            "query_length_chars": query_length,
            "agent_used": agent_used,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.logger.info(
            f"Query performance: {query_type} completed in {execution_time:.2f}s",
            extra={"performance_metrics": metrics}
        )
    
    def log_agent_performance(
        self,
        agent_name: str,
        operation: str,
        execution_time: float,
        success: bool,
        additional_data: Optional[dict] = None
    ) -> None:
        """Log agent performance metrics."""
        
        metrics = {
            "agent_name": agent_name,
            "operation": operation,
            "execution_time_seconds": execution_time,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if additional_data:
            metrics.update(additional_data)
        
        self.logger.info(
            f"Agent performance: {agent_name}.{operation} completed in {execution_time:.2f}s",
            extra={"performance_metrics": metrics}
        )
    
    def log_workflow_performance(
        self,
        workflow_id: str,
        total_time: float,
        agents_used: list,
        success: bool,
        error_count: int = 0
    ) -> None:
        """Log complete workflow performance."""
        
        metrics = {
            "workflow_id": workflow_id,
            "total_execution_time_seconds": total_time,
            "agents_used": agents_used,
            "success": success,
            "error_count": error_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(
            f"Workflow performance: completed in {total_time:.2f}s with {len(agents_used)} agents",
            extra={"performance_metrics": metrics}
        )


class SQLQueryLogger:
    """Specialized logger for SQL queries."""
    
    def __init__(self):
        self.logger = logging.getLogger("sql_queries")
    
    def log_query_review(
        self,
        query_id: str,
        original_query: str,
        corrected_query: str,
        violations: list,
        is_compliant: bool
    ) -> None:
        """Log SQL query review results."""
        
        self.logger.info(
            f"SQL Review - Query {query_id} - Compliant: {is_compliant}",
            extra={
                "query_id": query_id,
                "query_type": "review",
                "original_query": original_query[:500] + "..." if len(original_query) > 500 else original_query,
                "corrected_query": corrected_query[:500] + "..." if len(corrected_query) > 500 else corrected_query,
                "violations_count": len(violations),
                "violations": violations,
                "is_compliant": is_compliant
            }
        )
    
    def log_explain_generation(
        self,
        query_id: str,
        query: str,
        success: bool,
        execution_time: float,
        error_message: Optional[str] = None
    ) -> None:
        """Log EXPLAIN plan generation."""
        
        self.logger.info(
            f"EXPLAIN Generation - Query {query_id} - Success: {success}",
            extra={
                "query_id": query_id,
                "query_type": "explain",
                "query": query[:500] + "..." if len(query) > 500 else query,
                "success": success,
                "execution_time": execution_time,
                "error_message": error_message
            }
        )
    
    def log_performance_analysis(
        self,
        query_id: str,
        query: str,
        performance_rating: str,
        suggestions_count: int,
        critical_issues: int
    ) -> None:
        """Log performance analysis results."""
        
        self.logger.info(
            f"Performance Analysis - Query {query_id} - Rating: {performance_rating}",
            extra={
                "query_id": query_id,
                "query_type": "analysis",
                "query": query[:500] + "..." if len(query) > 500 else query,
                "performance_rating": performance_rating,
                "suggestions_count": suggestions_count,
                "critical_issues": critical_issues
            }
        )
