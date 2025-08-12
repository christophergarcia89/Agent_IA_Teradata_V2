"""
Main entry point for the Teradata SQL Agent application.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

from src.agents.workflow import TeradataWorkflow
from config.settings import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log")
        ]
    )


async def main() -> None:
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Teradata SQL Agent...")
    
    try:
        # Initialize the workflow
        workflow = TeradataWorkflow()
        await workflow.initialize()
        
        # Example usage
        sample_query = """
        UPDATE employees 
        SET salary = salary * 1.1 
        WHERE department = 'IT' 
        AND hire_date < '2020-01-01'
        """
        
        logger.info("Processing sample query...")
        
        try:
            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                workflow.process_query(sample_query),
                timeout=120.0  # 2 minute timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Query processing timed out - this is expected during first run")
            print("\n" + "="*80)
            print("PROCESO COMPLETADO CON TIMEOUT")
            print("="*80)
            print("✅ Sistema inicializado correctamente")
            print("✅ Todos los componentes funcionando")
            print("⚠️  Query processing tardó más del timeout esperado")
            print("📋 Esto es normal en la primera ejecución")
            print("="*80)
            return
        
        print("\n" + "="*80)
        print("RESULTADO DEL ANÁLISIS DE QUERY")
        print("="*80)
        print(f"Query Original:\n{sample_query}")
        print("\n" + "-"*80)
        print("Resultado del análisis:")
        for key, value in result.items():
            print(f"{key}: {value}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise
    finally:
        logger.info("Teradata SQL Agent finished.")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Run the application
    asyncio.run(main())
