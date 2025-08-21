"""
Validador Independiente de ChromaDB
===================================
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Añadir el directorio raíz al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vector_store import VectorStore
from src.utils.logging_utils import setup_logging

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

class ChromaDBValidator:
    """Validador independiente para ChromaDB y VectorStore."""
    
    def __init__(self):
        self.vector_store: Optional[VectorStore] = None
        self.validation_results: Dict[str, Any] = {}
        
    async def validate_chromadb_connection(self) -> bool:
        """Valida la conexión básica a ChromaDB."""
        try:
            logger.info("[VALIDATION] Iniciando validación de conexión ChromaDB...")
            
            # Crear instancia de VectorStore
            self.vector_store = VectorStore()
            logger.info("[SUCCESS] VectorStore instanciado correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Fallo en conexión ChromaDB: {e}")
            self.validation_results['connection_error'] = str(e)
            return False
    
    async def validate_initialization(self) -> bool:
        """Valida la inicialización completa del VectorStore."""
        try:
            logger.info("[VALIDATION] Iniciando validación de inicialización...")
            start_time = time.time()
            
            # Inicializar VectorStore
            await self.vector_store.initialize()
            init_time = time.time() - start_time
            
            logger.info(f"[SUCCESS] VectorStore inicializado en {init_time:.2f} segundos")
            self.validation_results['initialization_time'] = init_time
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Fallo en inicialización: {e}")
            self.validation_results['initialization_error'] = str(e)
            return False
    
    async def validate_document_loading(self) -> bool:
        """Valida la carga de documentos en ChromaDB y los carga si están faltando."""
        try:
            logger.info("[VALIDATION] Validando carga de documentos...")
            
            # Verificar si hay documentos cargados
            doc_count = await self.vector_store._get_collection_count()
            logger.info(f"[INFO] Documentos encontrados en ChromaDB: {doc_count}")
            
            self.validation_results['document_count'] = doc_count
            
            if doc_count > 0:
                logger.info("[SUCCESS] Documentos cargados correctamente en ChromaDB")
                return True
            else:
                logger.warning("[WARNING] No hay documentos en ChromaDB - Iniciando carga...")
                return await self._force_document_loading()
                
        except Exception as e:
            logger.error(f"[ERROR] Fallo en validación de documentos: {e}")
            self.validation_results['document_loading_error'] = str(e)
            return False
    
    async def _force_document_loading(self) -> bool:
        """Fuerza la carga completa de documentos desde knowledge_base."""
        try:
            logger.info("[LOADING] Iniciando carga forzada de documentos...")
            
            # Usar el método de carga de knowledge base del VectorStore
            await self.vector_store.load_knowledge_base()
            
            # Esperar un poco para que se complete el procesamiento
            await asyncio.sleep(2)
            
            # Verificar nuevamente el conteo
            doc_count = await self.vector_store._get_collection_count()
            logger.info(f"[INFO] Documentos después de la carga: {doc_count}")
            
            if doc_count > 0:
                logger.info(f"[SUCCESS] Carga forzada exitosa: {doc_count} documentos cargados")
                self.validation_results['document_count'] = doc_count
                self.validation_results['forced_loading'] = True
                return True
            else:
                logger.error("[ERROR] Carga forzada falló - No se cargaron documentos")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error en carga forzada: {e}")
            self.validation_results['forced_loading_error'] = str(e)
            return False
    
    async def validate_search_operations(self) -> bool:
        """Valida las operaciones de búsqueda en ChromaDB."""
        try:
            logger.info("[VALIDATION] Validando operaciones de búsqueda...")
            
            # Realizar búsqueda de prueba
            test_query = "SELECT * FROM tabla"
            start_time = time.time()
            
            # Usar el método correcto de búsqueda
            results = await self.vector_store.search_similar_examples(
                query=test_query,
                k=3
            )
            
            search_time = time.time() - start_time
            
            logger.info(f"[INFO] Búsqueda completada en {search_time:.3f} segundos")
            logger.info(f"[INFO] Resultados encontrados: {len(results)}")
            
            self.validation_results['search_time'] = search_time
            self.validation_results['search_results_count'] = len(results)
            
            if len(results) > 0:
                logger.info("[SUCCESS] Operaciones de búsqueda funcionando correctamente")
                return True
            else:
                logger.warning("[WARNING] Búsqueda no devolvió resultados")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Fallo en operaciones de búsqueda: {e}")
            self.validation_results['search_error'] = str(e)
            return False
    
    async def validate_vector_operations(self) -> bool:
        """Valida las operaciones vectoriales de ChromaDB."""
        try:
            logger.info("[VALIDATION] Validando operaciones vectoriales...")
            
            # Verificar que el modelo de embeddings funciona
            test_text = "CREATE TABLE test (id INT, name VARCHAR(50))"
            start_time = time.time()
            
            # Forzar la carga del modelo si no está disponible
            corporate_store = self.vector_store._corporate_store
            if corporate_store._embedding_model is None:
                corporate_store._ensure_embedding_model()
            
            # Verificar que el modelo de embeddings está disponible
            if hasattr(corporate_store, '_embedding_model') and corporate_store._embedding_model:
                # Crear embeddings usando el modelo interno
                embeddings = corporate_store._embedding_model.encode([test_text])
                embedding_time = time.time() - start_time
                
                if embeddings is not None and len(embeddings) > 0:
                    embedding_dim = len(embeddings[0]) if embeddings[0] is not None else 0
                    logger.info(f"[SUCCESS] Embeddings creados correctamente")
                    logger.info(f"[INFO] Dimensión de embeddings: {embedding_dim}")
                    logger.info(f"[INFO] Tiempo de creación: {embedding_time:.3f} segundos")
                    
                    self.validation_results['embedding_dimension'] = embedding_dim
                    self.validation_results['embedding_time'] = embedding_time
                    return True
                else:
                    logger.error("[ERROR] No se pudieron crear embeddings")
                    return False
            else:
                logger.warning("[WARNING] Modelo de embeddings no disponible")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Fallo en operaciones vectoriales: {e}")
            self.validation_results['vector_operations_error'] = str(e)
            return False
    
    async def validate_knowledge_base(self) -> bool:
        """Valida que la base de conocimiento esté cargada."""
        try:
            logger.info("[VALIDATION] Validando base de conocimiento...")
            
            # Verificar archivos de knowledge base
            kb_path = Path("knowledge_base")
            if not kb_path.exists():
                logger.error("[ERROR] Directorio knowledge_base no encontrado")
                return False
            
            # Contar archivos en knowledge_base
            doc_files = list(kb_path.rglob("*.txt"))
            logger.info(f"[INFO] Archivos de documentación encontrados: {len(doc_files)}")
            
            self.validation_results['kb_files_count'] = len(doc_files)
            
            # Verificar que los documentos están cargados usando el método correcto
            if hasattr(self.vector_store, 'get_stats'):
                try:
                    stats = self.vector_store.get_stats()  # Método síncrono
                    logger.info(f"[INFO] Estadísticas del VectorStore: {stats}")
                    self.validation_results['vector_store_stats'] = stats
                except Exception as stats_error:
                    logger.warning(f"[WARNING] No se pudieron obtener estadísticas: {stats_error}")
            
            # Verificar conteo de documentos
            doc_count = await self.vector_store._get_collection_count()
            if doc_count > 0:
                logger.info(f"[SUCCESS] Base de conocimiento cargada con {doc_count} documentos")
                return True
            else:
                logger.warning("[WARNING] Base de conocimiento vacía")
                return False
            
        except Exception as e:
            logger.error(f"[ERROR] Fallo en validación de base de conocimiento: {e}")
            self.validation_results['knowledge_base_error'] = str(e)
            return False
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Ejecuta validación completa de ChromaDB."""
        logger.info("="*60)
        logger.info("INICIANDO VALIDACIÓN COMPLETA DE CHROMADB")
        logger.info("="*60)
        
        validation_start = time.time()
        
        # Diccionario de validaciones
        validations = {
            'connection': self.validate_chromadb_connection,
            'initialization': self.validate_initialization,
            'document_loading': self.validate_document_loading,
            'search_operations': self.validate_search_operations,
            'vector_operations': self.validate_vector_operations,
            'knowledge_base': self.validate_knowledge_base
        }
        
        results = {}
        
        for validation_name, validation_func in validations.items():
            logger.info(f"\n[STEP] Ejecutando validación: {validation_name}")
            try:
                success = await validation_func()
                results[validation_name] = {
                    'success': success,
                    'timestamp': time.time()
                }
                status = "SUCCESS" if success else "FAILED"
                logger.info(f"[{status}] Validación {validation_name}: {status}")
                
            except Exception as e:
                logger.error(f"[ERROR] Excepción en validación {validation_name}: {e}")
                results[validation_name] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        total_time = time.time() - validation_start
        
        # Resumen final
        successful_validations = sum(1 for r in results.values() if r.get('success', False))
        total_validations = len(results)
        
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE VALIDACIÓN CHROMADB")
        logger.info("="*60)
        logger.info(f"Validaciones exitosas: {successful_validations}/{total_validations}")
        logger.info(f"Tiempo total: {total_time:.2f} segundos")
        
        if successful_validations == total_validations:
            logger.info("[SUCCESS] CHROMADB COMPLETAMENTE OPERATIVO")
        else:
            logger.warning(f"[WARNING] {total_validations - successful_validations} validaciones fallaron")
        
        # Compilar resultados finales
        final_results = {
            'validation_summary': {
                'total_validations': total_validations,
                'successful_validations': successful_validations,
                'success_rate': successful_validations / total_validations * 100,
                'total_time': total_time,
                'timestamp': time.time()
            },
            'individual_results': results,
            'detailed_metrics': self.validation_results
        }
        
        return final_results
    
    def print_detailed_report(self, results: Dict[str, Any]):
        """Imprime un reporte detallado de la validación."""
        print("\n" + "="*80)
        print("REPORTE DETALLADO DE VALIDACIÓN CHROMADB")
        print("="*80)
        
        summary = results['validation_summary']
        print(f"📊 Tasa de éxito: {summary['success_rate']:.1f}%")
        print(f"⏱️  Tiempo total: {summary['total_time']:.2f} segundos")
        
        print("\n📋 RESULTADOS INDIVIDUALES:")
        print("-" * 50)
        
        for validation_name, result in results['individual_results'].items():
            status = "SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{status} {validation_name.replace('_', ' ').title()}")
            
            if not result['success'] and 'error' in result:
                print(f"   Error: {result['error']}")
        
        print("\n📈 MÉTRICAS DETALLADAS:")
        print("-" * 50)
        
        metrics = results['detailed_metrics']
        if 'document_count' in metrics:
            print(f"Documentos en ChromaDB: {metrics['document_count']}")
        
        if 'initialization_time' in metrics:
            print(f"Tiempo de inicialización: {metrics['initialization_time']:.2f}s")
        
        if 'search_time' in metrics:
            print(f"Tiempo de búsqueda: {metrics['search_time']:.3f}s")
        
        if 'embedding_dimension' in metrics:
            print(f"Dimensión de embeddings: {metrics['embedding_dimension']}")
        
        if 'kb_files_count' in metrics:
            print(f"Archivos en knowledge base: {metrics['kb_files_count']}")

async def main():
    """Función principal para ejecutar la validación."""
    try:
        validator = ChromaDBValidator()
        results = await validator.run_full_validation()
        validator.print_detailed_report(results)
        
        return results['validation_summary']['success_rate'] == 100.0
        
    except Exception as e:
        logger.error(f"[FATAL] Error en validación principal: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando Validador Independiente de ChromaDB...")
    success = asyncio.run(main())
    
    if success:
        print("\n VALIDACIÓN COMPLETADA EXITOSAMENTE")
        exit(0)
    else:
        print("\n VALIDACIÓN COMPLETADA CON ERRORES")
        exit(1)