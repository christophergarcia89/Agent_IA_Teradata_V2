"""
Vector store implementation for RAG system.
Handles document embedding and similarity search.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from config.settings import settings
from src.rag.document_loader import DocumentLoader, SQLExample


class VectorStore:
    """Vector store for SQL examples and documentation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # NO inicializar embeddings inmediatamente - hacerlo lazy
        self._embeddings = None
        self._embedding_model = settings.embedding_model
        self.vectorstore: Optional[Chroma] = None
        self.persist_directory = Path(settings.chroma_persist_directory)
        self._embedding_cache = {}
        
    def clear_embedding_cache(self):
        """Clear embedding cache and force reload."""
        self._embeddings = None
        self._embedding_cache.clear()
        import gc
        gc.collect()
    
    def _create_embedding_safe_mode_1(self):
        """Create embeddings with special handling for meta tensor issues."""
        try:
            # Import here to avoid global dependency issues
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Force eager loading to avoid lazy/meta tensor issues
            model = SentenceTransformer(
                self._embedding_model,
                device='cpu',
                cache_folder=None,  # Force local loading
                use_auth_token=False  # Disable auth to avoid network issues
            )
            
            # Manually move to CPU and ensure materialization
            if hasattr(model, 'to'):
                try:
                    # Try the new recommended method first
                    model = model.to_empty(device='cpu')
                except (AttributeError, Exception):
                    # Fallback to regular method
                    model = model.to('cpu')
            
            # Wrap in LangChain HuggingFace embeddings
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=self._embedding_model,
                model=model,  # Use our pre-loaded model
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
        except Exception as safe_error:
            self.logger.warning(f"Safe mode 1 failed: {safe_error}")
            # Fallback to basic creation
            return HuggingFaceEmbeddings(
                model_name=self._embedding_model,
                model_kwargs={'device': 'cpu', 'trust_remote_code': False}
            )
        
    @property
    def embeddings(self):
        """Lazy loading of embeddings with robust error handling."""
        if self._embeddings is None:
            self.logger.info(f"[LOADING] Loading embedding model: {self._embedding_model}")
            self.logger.info("[WAIT] This may take a few minutes on first run...")
            
            # Try multiple approaches to handle PyTorch tensor issues
            attempts = [
                # Approach 1: Try to avoid the meta tensor issue completely
                lambda: self._create_embedding_safe_mode_1(),
                # Approach 2: Basic CPU configuration (most reliable)
                lambda: HuggingFaceEmbeddings(
                    model_name=self._embedding_model,
                    model_kwargs={'device': 'cpu', 'torch_dtype': 'float32'},
                    encode_kwargs={'normalize_embeddings': True}
                ),
                # Approach 3: Even more basic configuration
                lambda: HuggingFaceEmbeddings(
                    model_name=self._embedding_model,
                    model_kwargs={'device': 'cpu'}
                ),
                # Approach 4: Alternative model as fallback
                lambda: HuggingFaceEmbeddings(
                    model_name="jinaai/jina-embeddings-v2-base-code",
                    model_kwargs={'device': 'cpu'}
                )
            ]
            
            for i, approach in enumerate(attempts, 1):
                try:
                    self.logger.info(f"[ATTEMPT {i}] Trying approach {i}/4...")
                    
                    # Force garbage collection before attempting
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self._embeddings = approach()
                    
                    # Test the embedding to ensure it works
                    test_embedding = self._embeddings.embed_query("test")
                    if test_embedding and len(test_embedding) > 0:
                        self.logger.info(f"[SUCCESS] Embedding model loaded successfully (approach {i})")
                        break
                    else:
                        self.logger.warning(f"[WARNING] Approach {i} loaded but test failed")
                        self._embeddings = None
                        
                except Exception as e:
                    self.logger.warning(f"[FAILED] Approach {i} failed: {str(e)[:100]}...")
                    self._embeddings = None
                    
                    if i == len(attempts):
                        self.logger.error(f"[CRITICAL] All embedding approaches failed")
                        raise RuntimeError(f"Cannot load embedding model after {len(attempts)} attempts. Last error: {e}")
            
        return self._embeddings
        
    async def initialize(self) -> None:
        """Initialize the vector store with fast startup and progressive loading."""
        try:
            # Create persist directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.logger.info("[CHROMA] Starting fast ChromaDB initialization...")
            
            # FAST INITIALIZATION: Try to load existing collection first
            try:
                # First, try to connect to existing collection (should be fast)
                self.logger.info("[CHROMA] Attempting to connect to existing collection...")
                
                def create_chroma_fast():
                    # Use existing collection if available
                    return Chroma(
                        persist_directory=str(self.persist_directory),
                        embedding_function=self.embeddings,
                        collection_name="teradata_sql_knowledge"
                    )
                
                loop = asyncio.get_event_loop()
                self.vectorstore = await asyncio.wait_for(
                    loop.run_in_executor(None, create_chroma_fast),
                    timeout=30.0  # Reduced timeout for existing collection
                )
                
                # Quick check if collection has documents
                doc_count = await self._get_collection_count()
                
                if doc_count > 0:
                    self.logger.info(f"[CHROMA] Found existing collection with {doc_count} documents")
                    self.logger.info("[CHROMA] ChromaDB ready with existing data!")
                    return
                else:
                    self.logger.info("[CHROMA] Collection exists but is empty - will load documents")
                
            except asyncio.TimeoutError:
                self.logger.warning("[CHROMA] Timeout connecting to existing collection")
                # Create new in-memory collection for immediate use
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="teradata_sql_knowledge_memory"
                )
                self.logger.info("[CHROMA] Created in-memory collection as fallback")
            
            except Exception as init_error:
                self.logger.warning(f"[CHROMA] Error with existing collection: {init_error}")
                # Create new collection
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings,
                    collection_name="teradata_sql_knowledge"
                )
                self.logger.info("[CHROMA] Created new collection")
            
            # PROGRESSIVE LOADING: Start document loading in background immediately
            self.logger.info("[CHROMA] Starting immediate background document loading...")
            asyncio.create_task(self._fast_progressive_loading())
            
            self.logger.info("[CHROMA] [SUCCESS] Vector store initialized and ready!")
            self.logger.info("[CHROMA] [LOADING] Documents loading progressively in background...")
                
        except Exception as e:
            self.logger.error(f"[CHROMA] Error initializing vector store: {e}")
            # Create basic fallback
            try:
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="teradata_sql_fallback"
                )
                self.logger.info("[CHROMA] Created basic fallback vector store")
            except Exception as fallback_error:
                self.logger.error(f"[CHROMA] Even fallback failed: {fallback_error}")
                raise
            
    async def load_knowledge_base(self) -> None:
        """Load all documents into the vector store SYNCHRONOUSLY for complete initialization."""
        try:
            self.logger.info("[DOCS] Starting COMPLETE document loading process...")
            from src.rag.document_loader import DocumentLoader
            loader = DocumentLoader()
            
            # Load SQL examples
            self.logger.info("[DOCS] Loading SQL examples...")
            sql_examples = loader.load_sql_examples()
            sql_documents = loader.create_documents_from_examples(sql_examples)
            self.logger.info(f"[DOCS] Loaded {len(sql_documents)} SQL example documents")
            
            # Load documentation
            self.logger.info("[DOCS] Loading documentation files...")
            doc_documents = loader.load_documentation()
            self.logger.info(f"[DOCS] Loaded {len(doc_documents)} documentation files")
            
            # Combine all documents
            all_documents = sql_documents + doc_documents
            
            if not all_documents:
                self.logger.warning("[DOCS] No documents found to load")
                return
                
            # Chunk documents
            self.logger.info("[DOCS] Chunking documents...")
            chunked_docs = loader.chunk_documents(
                all_documents,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            self.logger.info(f"[DOCS] Created {len(chunked_docs)} document chunks")
            
            # Add to vector store with robust batch processing
            self.logger.info("[DOCS] Adding documents to vector store (COMPLETE LOADING)...")
            await self._add_all_documents_complete(chunked_docs)
            
            # Verify final count
            final_count = await self._get_collection_count()
            self.logger.info(f"[DOCS] [COMPLETE] COMPLETE LOADING FINISHED: {final_count} documents in ChromaDB")
                
        except Exception as e:
            self.logger.error(f"[DOCS] Error in complete knowledge base loading: {e}")
            raise  # Re-raise to fail workflow initialization if documents can't be loaded
    
    async def _add_all_documents_complete(self, documents: List) -> None:
        """Add all documents with progress tracking and error handling."""
        try:
            total_docs = len(documents)
            batch_size = 50  # Larger batches for efficiency
            successful_batches = 0
            failed_batches = 0
            
            self.logger.info(f"[DOCS] Processing {total_docs} documents in batches of {batch_size}")
            
            # Process in batches with progress reporting
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_docs - 1) // batch_size + 1
                
                try:
                    # Add batch with extended timeout
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: self.vectorstore.add_documents(batch)
                        ),
                        timeout=60.0  # 60 seconds per batch
                    )
                    
                    successful_batches += 1
                    progress = (batch_num / total_batches) * 100
                    self.logger.info(f"[DOCS] [SUCCESS] Batch {batch_num}/{total_batches} ({progress:.1f}%) - {len(batch)} docs added")
                    
                    # Short pause between batches
                    await asyncio.sleep(0.5)
                    
                except asyncio.TimeoutError:
                    failed_batches += 1
                    self.logger.error(f"[DOCS] [ERROR] Batch {batch_num} timed out")
                    
                except Exception as batch_error:
                    failed_batches += 1
                    self.logger.error(f"[DOCS] [ERROR] Batch {batch_num} failed: {batch_error}")
            
            # Summary
            self.logger.info(f"[DOCS] Batch processing complete: {successful_batches} successful, {failed_batches} failed")
            
            if failed_batches > successful_batches:
                raise RuntimeError(f"Too many batch failures: {failed_batches}/{successful_batches + failed_batches}")
            
        except Exception as e:
            self.logger.error(f"[DOCS] Error in complete document addition: {e}")
            raise
    
    async def load_knowledge_base_incremental(self) -> None:
        """Load documents incrementally with better timeout handling."""
        try:
            self.logger.info("[DOCS] Starting incremental document loading...")
            from src.rag.document_loader import DocumentLoader
            loader = DocumentLoader()
            
            # Load documents in smaller batches
            self.logger.info("[DOCS] Loading SQL examples...")
            sql_examples = loader.load_sql_examples()
            
            if sql_examples:
                # Convert SQL examples to LangChain Documents
                sql_docs = loader.create_documents_from_examples(sql_examples)
                await self._add_documents_batch(sql_docs, "SQL examples", batch_size=10)
            
            self.logger.info("[DOCS] Loading documentation files...")
            doc_files = loader.load_documentation()
            
            if doc_files:
                # Convertir documentos con metadatos específicos
                for doc in doc_files:
                    doc.metadata.update({
                        "type": "documentation",
                        "doc_type": "documentation",
                        "source_type": "documentation"
                    })
                await self._add_documents_batch(doc_files, "documentation", batch_size=5)
                
            self.logger.info("[DOCS] Incremental loading completed successfully")
            
        except Exception as e:
            self.logger.error(f"[DOCS] Error in incremental loading: {e}")
    
    async def _fast_progressive_loading(self) -> None:
        """Fast progressive loading optimized for immediate use."""
        try:
            self.logger.info("[DOCS] [LAUNCH] Starting FAST progressive loading...")
            
            # Small delay to let initialization complete
            await asyncio.sleep(1.0)
            
            from src.rag.document_loader import DocumentLoader
            loader = DocumentLoader()
            
            # PHASE 1: Load critical SQL examples first (small batch)
            self.logger.info("[DOCS] PHASE 1: Loading critical SQL examples...")
            
            try:
                sql_examples = loader.load_sql_examples()
                
                if sql_examples:
                    # Load only first 10 examples for immediate availability
                    critical_examples = sql_examples[:10]  
                    sql_docs = loader.create_documents_from_examples(critical_examples)
                    
                    # Add immediately with short timeout
                    await self._add_documents_fast(sql_docs, "critical_examples", batch_size=5)
                    self.logger.info(f"[DOCS] [SUCCESS] Loaded {len(critical_examples)} critical examples")
                
                    # PHASE 2: Load remaining examples in background
                    if len(sql_examples) > 10:
                        remaining_examples = sql_examples[10:]
                        asyncio.create_task(self._load_remaining_documents(remaining_examples, loader))
                        
            except Exception as phase1_error:
                self.logger.error(f"[DOCS] Phase 1 error: {phase1_error}")
            
            # PHASE 3: Load documentation (low priority)
            try:
                self.logger.info("[DOCS] PHASE 3: Loading documentation...")
                doc_files = loader.load_documentation()
                
                if doc_files:
                    asyncio.create_task(self._add_documents_fast(doc_files, "documentation", batch_size=2))
                    
            except Exception as phase3_error:
                self.logger.error(f"[DOCS] Phase 3 error: {phase3_error}")
            
            self.logger.info("[DOCS] [SUCCESS] Fast progressive loading started successfully")
            
        except Exception as e:
            self.logger.error(f"[DOCS] Fast progressive loading error: {e}")
    
    async def _load_remaining_documents(self, remaining_examples: list, loader) -> None:
        """Load remaining documents in background with chunking."""
        try:
            self.logger.info(f"[DOCS] Loading remaining {len(remaining_examples)} examples in background...")
            
            # Process in chunks of 20
            chunk_size = 20
            for i in range(0, len(remaining_examples), chunk_size):
                chunk = remaining_examples[i:i + chunk_size]
                sql_docs = loader.create_documents_from_examples(chunk)
                
                await self._add_documents_fast(sql_docs, f"batch_{i//chunk_size + 1}", batch_size=10)
                
                # Progress report
                processed = min(i + chunk_size, len(remaining_examples))
                progress = (processed / len(remaining_examples)) * 100
                self.logger.info(f"[DOCS] [PROGRESS] Progress: {processed}/{len(remaining_examples)} ({progress:.1f}%)")
                
                # Short break between chunks
                await asyncio.sleep(0.5)
            
            self.logger.info("[DOCS] [SUCCESS] All documents loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"[DOCS] Error loading remaining documents: {e}")
    
    async def _add_documents_fast(self, documents: List, doc_type: str, batch_size: int = 5) -> None:
        """Add documents with optimized timeout and error handling."""
        try:
            from src.rag.document_loader import DocumentLoader
            loader = DocumentLoader()
            
            # Chunk documents with smaller chunks for faster processing
            chunked_docs = loader.chunk_documents(
                documents,
                chunk_size=500,  # Smaller chunks
                chunk_overlap=100  # Less overlap
            )
            
            self.logger.info(f"[DOCS] Adding {len(chunked_docs)} {doc_type} chunks (batch_size: {batch_size})")
            
            # Process in very small batches with aggressive timeouts
            for i in range(0, len(chunked_docs), batch_size):
                batch = chunked_docs[i:i + batch_size]
                
                try:
                    # Ultra-fast timeout for each batch
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: self.vectorstore.add_documents(batch)
                        ),
                        timeout=15.0  # 15 seconds per batch
                    )
                    
                    batch_num = i//batch_size + 1
                    total_batches = (len(chunked_docs)-1)//batch_size + 1
                    self.logger.info(f"[DOCS] [SUCCESS] Batch {batch_num}/{total_batches} added")
                    
                    # Micro break
                    await asyncio.sleep(0.1)
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"[DOCS] [WARNING] Timeout on batch {i//batch_size + 1} of {doc_type}")
                    continue
                except Exception as batch_error:
                    self.logger.warning(f"[DOCS] [WARNING] Error on batch {i//batch_size + 1}: {batch_error}")
                    continue
            
            self.logger.info(f"[DOCS] [SUCCESS] Completed {doc_type} loading")
            
        except Exception as e:
            self.logger.error(f"[DOCS] Error in fast document addition for {doc_type}: {e}")

    async def load_knowledge_base_background(self) -> None:
        """Load documents in background without blocking initialization."""
        try:
            self.logger.info("[DOCS] Starting background document loading...")
            
            # Add a small delay to let initialization complete
            await asyncio.sleep(2.0)
            
            # Load with longer timeout since it's background
            await asyncio.wait_for(
                self.load_knowledge_base_incremental(),
                timeout=600.0  # 10 minute timeout for background loading
            )
            
            self.logger.info("[DOCS] Background loading completed")
            
        except asyncio.TimeoutError:
            self.logger.warning("[DOCS] Background loading timed out - partial documents may be available")
        except Exception as e:
            self.logger.error(f"[DOCS] Error in background loading: {e}")
    
    async def _add_documents_batch(self, documents: List, doc_type: str, batch_size: int = 10) -> None:
        """Add documents in batches to avoid timeout."""
        try:
            from src.rag.document_loader import DocumentLoader
            loader = DocumentLoader()
            
            # Chunk documents
            chunked_docs = loader.chunk_documents(
                documents,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            self.logger.info(f"[DOCS] Adding {len(chunked_docs)} {doc_type} chunks in batches of {batch_size}")
            
            # Process in batches
            for i in range(0, len(chunked_docs), batch_size):
                batch = chunked_docs[i:i + batch_size]
                
                try:
                    # Add batch with timeout
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: self.vectorstore.add_documents(batch)
                        ),
                        timeout=30.0  # 30 seconds per batch
                    )
                    
                    self.logger.info(f"[DOCS] Added batch {i//batch_size + 1}/{(len(chunked_docs)-1)//batch_size + 1} ({len(batch)} docs)")
                    
                    # Small delay between batches
                    await asyncio.sleep(0.5)
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"[DOCS] Timeout adding batch {i//batch_size + 1} of {doc_type}")
                    continue
                except Exception as batch_error:
                    self.logger.error(f"[DOCS] Error adding batch {i//batch_size + 1}: {batch_error}")
                    continue
            
            self.logger.info(f"[DOCS] Completed adding {doc_type} documents")
            
        except Exception as e:
            self.logger.error(f"[DOCS] Error in batch processing for {doc_type}: {e}")
    
    async def search_similar_examples(self, query: str, k: int = None) -> List[Document]:
        """Search for similar SQL examples."""
        if not self.vectorstore:
            self.logger.warning("Vector store not initialized - returning empty results")
            return []
            
        k = k or settings.top_k_retrieval
        
        try:
            # Ultra-safe search with very short timeout
            results = await asyncio.wait_for(
                self._async_safe_search(query, k, {"type": "sql_example"}),
                timeout=2.0  # Very short timeout
            )
            
            self.logger.debug(f"Found {len(results)} similar examples for query")
            return results
            
        except asyncio.TimeoutError:
            self.logger.info(f"Search timeout for similar examples - documents may still be loading")
            return []
        except Exception as e:
            self.logger.warning(f"Error searching similar examples: {e}")
            return []
    
    async def search_documentation(self, query: str, k: int = None) -> List[Document]:
        """Search for relevant documentation with enhanced filters."""
        if not self.vectorstore:
            self.logger.warning("Vector store not initialized - returning empty results")
            return []
            
        k = k or settings.top_k_retrieval
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                attempt += 1
                self.logger.info(f"[SEARCH] Documentation search attempt {attempt}/{max_attempts}")
                
                # Buscar primero sin filtros pero con más resultados
                results = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.vectorstore.similarity_search(query, k=k*3)  # Buscar más resultados para filtrar
                    ),
                    timeout=5.0
                )
                
                # Filtrar resultados basados en la fuente o metadata
                filtered_results = []
                for doc in results:
                    source = str(doc.metadata.get("source", "")).lower()
                    if "documentation" in source or "normativa" in source:
                        filtered_results.append(doc)
                        continue
                        
                    # Verificar metadata
                    metadata = doc.metadata
                    if any(metadata.get(key) == "documentation" 
                          for key in ['type', 'doc_type', 'source_type', 'document_type', 'category']):
                        filtered_results.append(doc)
                
                # Si encontramos resultados, devolver los k mejores
                if filtered_results:
                    filtered_results = filtered_results[:k]  # Limitar al número solicitado
                    self.logger.info(f"[SEARCH] Found {len(filtered_results)} documentation documents")
                    return filtered_results
                
                # Si no hay resultados, intentar búsqueda más específica
                try:
                    # Usar múltiples queries relacionadas
                    queries = [
                        query,
                        f"{query} sql",
                        f"{query} teradata",
                        f"{query} standard",
                        f"{query} normativa"
                    ]
                    
                    for search_query in queries:
                        results = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self._safe_similarity_search(
                                    search_query, 
                                    k,
                                    {"type": "documentation"}
                                )
                            ),
                            timeout=3.0
                        )
                        if results:
                            self.logger.info(f"[SEARCH] Found {len(results)} docs with query: {search_query}")
                            return results
                            
                except Exception as search_error:
                    self.logger.warning(f"[SEARCH] Search error: {search_error}")
                
                self.logger.warning(f"[SEARCH] No results in attempt {attempt}, retrying...")
                await asyncio.sleep(1)
                
            except asyncio.TimeoutError:
                self.logger.warning(f"[SEARCH] Timeout in attempt {attempt}")
                if attempt < max_attempts:
                    await asyncio.sleep(1)
                continue
            except Exception as e:
                self.logger.error(f"[SEARCH] Error in attempt {attempt}: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(1)
                continue
        
        self.logger.warning("[SEARCH] All search attempts failed - returning empty")
        return []
    
    async def search_ok_examples_by_category(self, category: str, k: int = 5) -> List[Document]:
        """Search for OK examples in a specific category."""
        if not self.vectorstore:
            self.logger.warning("Vector store not initialized - returning empty results")
            return []
            
        try:
            # Simple timeout-protected search
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._safe_similarity_search(
                        f"{category} SQL best practices",
                        k,
                        {
                            "type": "sql_example",
                            "example_type": "OK",
                            "category": category
                        }
                    )
                ),
                timeout=5.0  # Reduced timeout
            )
            
            self.logger.debug(f"Found {len(results)} OK examples for category {category}")
            return results
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Search timeout for OK examples in category {category} - returning empty results")
            return []
        except Exception as e:
            self.logger.error(f"Error searching OK examples for category {category}: {e}")
            return []
    
    async def search_with_score(self, query: str, k: int = None, 
                               score_threshold: float = None) -> List[tuple]:
        """Search with similarity scores."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
            
        k = k or settings.top_k_retrieval
        score_threshold = score_threshold or settings.similarity_threshold
        
        try:
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            
            self.logger.debug(
                f"Found {len(filtered_results)} results above threshold {score_threshold}"
            )
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error searching with scores: {e}")
            return []
    
    async def get_related_pairs(self, query: str) -> Dict[str, List[Document]]:
        """Get related OK/NOK pairs for a query."""
        try:
            # Search for related examples
            all_results = await self.search_similar_examples(query, k=20)
            
            # Group by category and example number
            pairs = {}
            for doc in all_results:
                metadata = doc.metadata
                category = metadata.get("category", "unknown")
                example_num = metadata.get("example_number", "unknown")
                example_type = metadata.get("example_type", "unknown")
                
                key = f"{category}_{example_num}"
                if key not in pairs:
                    pairs[key] = {"OK": [], "NOK": []}
                    
                pairs[key][example_type].append(doc)
            
            # Filter to only complete pairs
            complete_pairs = {
                key: pair for key, pair in pairs.items()
                if pair["OK"] and pair["NOK"]
            }
            
            self.logger.debug(f"Found {len(complete_pairs)} complete OK/NOK pairs")
            return complete_pairs
            
        except Exception as e:
            self.logger.error(f"Error getting related pairs: {e}")
            return {}
    
    def reset_vector_store(self) -> None:
        """Reset the vector store (delete all data)."""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                
            # Remove persist directory
            if self.persist_directory.exists():
                import shutil
                shutil.rmtree(self.persist_directory)
                
            self.logger.info("Vector store reset successfully")
            
        except Exception as e:
            self.logger.error(f"Error resetting vector store: {e}")
            raise
    
    async def _get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            if not self.vectorstore:
                return 0
                
            # Use ChromaDB's count method with timeout protection
            count = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.vectorstore._collection.count()
                ),
                timeout=5.0  # 5 second timeout
            )
            return count
            
        except asyncio.TimeoutError:
            self.logger.warning("Timeout getting collection count - assuming empty")
            return 0
        except Exception as e:
            self.logger.warning(f"Error getting collection count: {e} - assuming empty")
            return 0
    
    def _safe_similarity_search(self, query: str, k: int, filter_dict: dict = None) -> List[Document]:
        """Perform similarity search with error handling and progressive loading awareness."""
        try:
            if not self.vectorstore:
                self.logger.info("[SEARCH] Vector store not ready - returning empty results")
                return []
            
            # Quick readiness check
            try:
                collection_name = self.vectorstore._collection.name
                if not collection_name:
                    return []
            except Exception:
                # Collection may still be initializing
                self.logger.info("[SEARCH] Collection still initializing - returning empty results")
                return []
            
            # Get document count with timeout protection
            try:
                count_future = asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.vectorstore._collection.count()
                )
                count = asyncio.get_event_loop().run_until_complete(
                    asyncio.wait_for(count_future, timeout=2.0)
                )
                
                if count == 0:
                    self.logger.info("[SEARCH] Collection empty - may still be loading documents")
                    return []
                    
                self.logger.debug(f"[SEARCH] Collection has {count} documents available")
                
            except (asyncio.TimeoutError, Exception):
                # If count check fails, try search anyway with reduced k
                self.logger.debug("[SEARCH] Count check failed - trying search with reduced results")
                k = min(k, 3)  # Reduce results for safety
            
            # Perform search with error recovery
            try:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
                
                self.logger.debug(f"[SEARCH] Found {len(results)} results for query")
                return results
                
            except Exception as search_error:
                # Fallback: try with minimal parameters
                self.logger.warning(f"[SEARCH] Primary search failed: {search_error}")
                try:
                    # Fallback search without filters
                    fallback_results = self.vectorstore.similarity_search(query, k=min(k, 2))
                    self.logger.info(f"[SEARCH] Fallback search returned {len(fallback_results)} results")
                    return fallback_results
                except Exception:
                    self.logger.warning("[SEARCH] Even fallback search failed - returning empty")
                    return []
            
        except Exception as e:
            self.logger.warning(f"[SEARCH] Safe similarity search failed: {e} - returning empty")
            return []
    
    async def _async_safe_search(self, query: str, k: int, filter_dict: dict = None) -> List[Document]:
        """Async version of safe search that can be cancelled quickly."""
        try:
            # First check if we can do a quick status check
            if not self.vectorstore:
                return []
            
            # Run the potentially blocking search in executor with very tight control
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._safe_similarity_search, 
                query, k, filter_dict
            )
            return result
            
        except Exception as e:
            self.logger.warning(f"Async safe search failed: {e}")
            return []