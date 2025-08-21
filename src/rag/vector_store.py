"""
Vector store
"""

import logging
import asyncio
import os
import ssl
import warnings
import gc
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import json
import torch

# Corporate SSL bypass
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from langchain.schema import Document

from config.settings import settings
from src.rag.document_loader import DocumentLoader, SQLExample


class CorporateVectorStore:
    """
    Corporate-safe vector store implementation with enhanced features.
    Uses local file storage and pre-computed embeddings to avoid ChromaDB issues.
    Includes progressive loading, batch processing, and robust error handling.
    """
    
    def __init__(
        self, 
        collection_name: str = "teradata_sql_knowledge",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory or settings.chroma_persist_directory)
        self._embedding_model_name = embedding_model or settings.embedding_model
        
        # Create directories
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.documents_file = self.persist_directory / f"{collection_name}_docs.json"
        self.embeddings_file = self.persist_directory / f"{collection_name}_embeddings.npy" 
        self.metadata_file = self.persist_directory / f"{collection_name}_metadata.json"
        
        # In-memory storage
        self._embedding_model = None
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Tracking and caching
        self._initialized = False
        self._loading_started = False
        self._embedding_cache = {}
        
        # Progressive loading configuration
        self.batch_size = 128
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.loading_chunk_size = 20  # Chunk size for background loading
    
    def clear_embedding_cache(self):
        """Clear embedding cache and force reload - from original best practices."""
        self._embedding_model = None
        self._embedding_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("[CACHE] Embedding cache cleared")
    
    def _create_embedding_safe_mode_corporate(self):
        """
        Corporate-safe embedding model creation with multiple fallback approaches.
        Combines original best practices with corporate SSL bypass.
        """
        # Try multiple approaches to handle PyTorch tensor issues
        attempts = [
            # Approach 1: Corporate-optimized primary model
            lambda: self._create_model_approach_1(),
            # Approach 2: Fallback with different configuration
            lambda: self._create_model_approach_2(), 
            # Approach 3: Alternative model
            lambda: self._create_model_approach_3(),
            # Approach 4: Ultra-simple fallback
            lambda: self._create_model_approach_4()
        ]
        
        for i, approach in enumerate(attempts, 1):
            try:
                self.logger.info(f"[EMBEDDING] Trying approach {i}/4...")
                
                # Force garbage collection before attempting
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                model = approach()
                
                # Test the model
                test_result = model.encode(["test"])
                if test_result is not None and len(test_result) > 0:
                    dimensions = len(test_result[0]) if hasattr(test_result[0], '__len__') else len(test_result)
                    self.logger.info(f"[EMBEDDING] SUCCESS Model loaded successfully (approach {i}) - Dimensions: {dimensions}")
                    return model
                else:
                    self.logger.warning(f"[EMBEDDING] WARNING Approach {i} loaded but test failed")
                    
            except Exception as e:
                self.logger.warning(f"[EMBEDDING] FAILED Approach {i} failed: {str(e)[:100]}...")
                
                if i == len(attempts):
                    self.logger.error(f"[EMBEDDING] FAILED All embedding approaches failed")
                    raise RuntimeError(f"Cannot load embedding model after {len(attempts)} attempts. Last error: {e}")
        
        return None
    
    def _create_model_approach_1(self):
        """Primary corporate approach with tensor handling."""
        model = SentenceTransformer(
            self._embedding_model_name,
            device='cpu',
            cache_folder=None,  # Force local loading
            use_auth_token=False  # Disable auth to avoid network issues
        )
        
        # Handle tensor materialization issues
        if hasattr(model, 'to'):
            try:
                model = model.to_empty(device='cpu')
            except (AttributeError, Exception):
                model = model.to('cpu')
        
        return model
    
    def _create_model_approach_2(self):
        """Fallback with explicit device configuration."""
        return SentenceTransformer(
            self._embedding_model_name,
            device='cpu'
        )
    
    def _create_model_approach_3(self):
        """Alternative model fallback."""
        return SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu'
        )
    
    def _create_model_approach_4(self):
        """Ultra-simple fallback."""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def _ensure_embedding_model(self):
        """Load embedding model on demand with robust error handling."""
        if self._embedding_model is None:
            self.logger.info(f"[EMBEDDING] MODEL Loading model: {self._embedding_model_name}")
            self.logger.info("[EMBEDDING] LOADING This may take a few minutes on first run...")
            
            self._embedding_model = self._create_embedding_safe_mode_corporate()
            
            if self._embedding_model is None:
                raise RuntimeError("Failed to load any embedding model")
            
            # Warm-up the model to ensure consistent results
            self._warmup_embedding_model()
    
    def _warmup_embedding_model(self):
        """Warm up the embedding model to ensure consistent search results."""
        try:
            self.logger.info("[EMBEDDING] WARMUP Warming up embedding model...")
            
            # Test embeddings with sample queries to initialize internal state
            warmup_queries = [
                "SELECT * FROM table",
                "UPDATE table SET column = value", 
                "CREATE TABLE example",
                "INSERT INTO table VALUES",
                "SQL examples"
            ]
            
            for query in warmup_queries:
                try:
                    embedding = self._embedding_model.encode([query])
                    if embedding is not None and len(embedding) > 0:
                        continue  # Success
                except Exception as e:
                    self.logger.warning(f"[EMBEDDING] WARMUP Warning during warmup: {e}")
            
            self.logger.info("[EMBEDDING] WARMUP Model warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"[EMBEDDING] WARMUP Warning: Warmup failed: {e}")
            # Don't fail initialization if warmup fails
    
    def _load_persisted_data(self):
        """Load persisted documents and embeddings with enhanced error handling."""
        try:
            loaded_count = 0
            
            # Load documents
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                loaded_count += len(self.documents)
            
            # Load embeddings
            if self.embeddings_file.exists():
                self.embeddings = np.load(self.embeddings_file)
                # Validate embeddings shape
                if len(self.documents) != len(self.embeddings):
                    self.logger.warning(f"[STORAGE] WARNING Mismatch: {len(self.documents)} docs vs {len(self.embeddings)} embeddings")
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            if loaded_count > 0:
                self.logger.info(f"[STORAGE] LOADED Loaded {loaded_count} persisted documents")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"[STORAGE] ERROR Error loading persisted data: {e}")
            self.documents = []
            self.embeddings = None
            self.metadata = []
            return False
    
    def _save_persisted_data(self):
        """Save documents and embeddings to disk with validation."""
        try:
            # Validate data consistency
            doc_count = len(self.documents)
            emb_count = len(self.embeddings) if self.embeddings is not None else 0
            meta_count = len(self.metadata)
            
            if doc_count != emb_count:
                self.logger.warning(f"[STORAGE] WARNING Data inconsistency: {doc_count} docs vs {emb_count} embeddings")
            
            # Save documents
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"[STORAGE] SAVED Saved {doc_count} documents, {emb_count} embeddings, {meta_count} metadata")
            
        except Exception as e:
            self.logger.error(f"[STORAGE] ERROR Error saving data: {e}")
            raise
    
    async def initialize(self) -> None:
        """
        Initialize the corporate vector store with fast startup and progressive loading.
        Enhanced version of original initialize method.
        """
        try:
            self.logger.info("[INIT] CORPORATE Starting Corporate Vector Store initialization...")
            
            # Quick check for existing data
            if self._load_persisted_data():
                doc_count = len(self.documents)
                self.logger.info(f"[INIT] SUCCESS Found existing collection with {doc_count} documents")
                self.logger.info("[INIT] READY Vector store ready with existing data!")
                self._initialized = True
                return
            
            # No existing data - will need to load
            self.logger.info("[INIT] FILE No existing data found")
            self.logger.info("[INIT] READY Vector store initialized - Documents will load progressively")
            
            # Start fast progressive loading
            if not self._loading_started:
                self._loading_started = True
                asyncio.create_task(self._fast_progressive_loading())
            
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"[INIT] ERROR Initialization error: {e}")
            raise
    
    async def _fast_progressive_loading(self):
        """
        Fast progressive loading with prioritized content.
        Enhanced version from original implementation.
        """
        try:
            self.logger.info("[LOADING] STARTING Starting FAST progressive loading...")
            
            # Small delay to let initialization complete
            await asyncio.sleep(1.0)
            
            loader = DocumentLoader()
            
            # PHASE 1: Load critical SQL examples first (small batch)
            self.logger.info("[LOADING] PHASE1 PHASE 1: Loading critical SQL examples...")
            
            try:
                sql_examples = loader.load_sql_examples()
                
                if sql_examples:
                    # Load only first 10 examples for immediate availability
                    critical_examples = sql_examples[:10]  
                    sql_docs = loader.create_documents_from_examples(critical_examples)
                    
                    # Add immediately with corporate-safe processing
                    await self._add_documents_fast(sql_docs, "critical_examples", batch_size=5)
                    self.logger.info(f"[LOADING] SUCCESS Loaded {len(critical_examples)} critical examples")
                
                    # PHASE 2: Load remaining examples in background
                    if len(sql_examples) > 10:
                        remaining_examples = sql_examples[10:]
                        asyncio.create_task(self._load_remaining_documents(remaining_examples, loader))
                        
            except Exception as phase1_error:
                self.logger.error(f"[LOADING] ERROR Phase 1 error: {phase1_error}")
            
            # PHASE 3: Load documentation (low priority)
            try:
                self.logger.info("[LOADING] PHASE3 PHASE 3: Loading documentation...")
                doc_files = loader.load_documentation()
                
                if doc_files:
                    asyncio.create_task(self._add_documents_fast(doc_files, "documentation", batch_size=2))
                    
            except Exception as phase3_error:
                self.logger.error(f"[LOADING] ERROR Phase 3 error: {phase3_error}")
            
            self.logger.info("[LOADING] SUCCESS Fast progressive loading started successfully")
            
        except Exception as e:
            self.logger.error(f"[LOADING] ERROR Fast progressive loading error: {e}")
    
    async def _load_remaining_documents(self, remaining_examples: list, loader) -> None:
        """Load remaining documents in background with chunking and enhanced error handling."""
        try:
            # Validate input parameters
            if not remaining_examples:
                self.logger.warning("[LOADING] WARNING No remaining examples to load")
                return
            
            if loader is None:
                self.logger.error("[LOADING] ERROR DocumentLoader is None")
                return
            
            self.logger.info(f"[LOADING] PROCESSING Loading {len(remaining_examples)} remaining examples...")
            
            # Use configurable chunk size (fallback to default if not set)
            chunk_size = getattr(self, 'loading_chunk_size', 20)
            total_chunks = (len(remaining_examples) + chunk_size - 1) // chunk_size
            
            # Process in chunks with progress tracking
            for i in range(0, len(remaining_examples), chunk_size):
                chunk_num = i // chunk_size + 1
                
                try:
                    chunk = remaining_examples[i:i + chunk_size]
                    
                    self.logger.info(f"[LOADING] CHUNK Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} examples)")
                    
                    # Create documents from chunk
                    sql_docs = loader.create_documents_from_examples(chunk)
                    
                    if sql_docs:
                        # Add documents with timeout protection
                        await asyncio.wait_for(
                            self._add_documents_fast(sql_docs, f"batch_{chunk_num}", batch_size=10),
                            timeout=60.0  # 60 seconds timeout per chunk
                        )
                        
                        self.logger.info(f"[LOADING] SUCCESS Chunk {chunk_num}/{total_chunks} completed")
                    else:
                        self.logger.warning(f"[LOADING] WARNING Chunk {chunk_num} produced no documents")
                    
                    # Memory cleanup and delay between chunks
                    if chunk_num % 5 == 0:  # Every 5 chunks
                        gc.collect()  # Force garbage collection
                    
                    # Progressive delay - shorter for first chunks, longer for later ones
                    delay = min(2.0 + (chunk_num * 0.1), 5.0)
                    await asyncio.sleep(delay)
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"[LOADING] TIMEOUT Chunk {chunk_num} timed out - skipping")
                    continue
                    
                except Exception as chunk_error:
                    self.logger.error(f"[LOADING] ERROR Chunk {chunk_num} failed: {chunk_error}")
                    # Continue with next chunk instead of failing completely
                    continue
            
            self.logger.info(f"[LOADING] SUCCESS Remaining documents loading completed ({total_chunks} chunks processed)")
            
        except Exception as e:
            self.logger.error(f"[LOADING] ERROR Critical error loading remaining documents: {e}")
            # Don't re-raise exception as this runs in background task
    
    async def _add_documents_fast(self, documents: List[Document], doc_type: str, batch_size: int = 5) -> None:
        """
        Add documents with optimized timeout and error handling.
        Corporate-safe version of original fast add method.
        """
        try:
            if not documents:
                return
            
            loader = DocumentLoader()
            
            # Chunk documents with smaller chunks for faster processing
            chunked_docs = loader.chunk_documents(
                documents,
                chunk_size=self.chunk_size,  # Configurable chunk size
                chunk_overlap=self.chunk_overlap  # Configurable overlap
            )
            
            self.logger.info(f"[ADD] ADDING Adding {len(chunked_docs)} {doc_type} chunks (batch_size: {batch_size})")
            
            # Process in small batches
            for i in range(0, len(chunked_docs), batch_size):
                batch = chunked_docs[i:i + batch_size]
                
                try:
                    # Add batch with timeout protection
                    await asyncio.wait_for(
                        self._process_batch_corporate_safe(batch),
                        timeout=30.0  # 30 seconds per batch
                    )
                    
                    self.logger.info(f"[ADD] SUCCESS Added batch {i//batch_size + 1}/{(len(chunked_docs) + batch_size - 1)//batch_size}")
                    
                    # Small delay between batches
                    await asyncio.sleep(1.0)
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"[ADD] Batch {i//batch_size + 1} timeout - skipping")
                    continue
                    
                except Exception as batch_error:
                    self.logger.warning(f"[ADD] ERROR Batch {i//batch_size + 1} error: {batch_error}")
                    continue
            
            self.logger.info(f"[ADD] COMPLETED Completed adding {doc_type} documents")
            
        except Exception as e:
            self.logger.error(f"[ADD] ERROR Error in fast add documents: {e}")
    
    async def _process_batch_corporate_safe(self, batch: List[Document]):
        """Process a batch of documents in corporate-safe manner."""
        # Prepare data
        doc_texts = [doc.page_content for doc in batch]
        doc_metadata = [doc.metadata for doc in batch]
        
        # Ensure embedding model is loaded
        self._ensure_embedding_model()
        
        # Calculate embeddings
        new_embeddings = self._embedding_model.encode(
            doc_texts, 
            show_progress_bar=False,  # Disable progress bar for batch processing
            batch_size=self.batch_size
        )
        
        # Add to storage
        self.documents.extend(doc_texts)
        self.metadata.extend(doc_metadata)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Persist data periodically
        if len(self.documents) % 50 == 0:  # Save every 50 documents
            self._save_persisted_data()
    
    async def load_knowledge_base(self) -> None:
        """
        Load all knowledge base documents with enhanced error handling.
        Enhanced version of original method.
        """
        try:
            self.logger.info("[DOCS] STARTING Starting knowledge base loading...")
            
            # Initialize document loader
            loader = DocumentLoader()
            
            # Load SQL examples
            self.logger.info("[DOCS] Loading SQL examples...")
            sql_examples = loader.load_sql_examples()
            sql_documents = loader.create_documents_from_examples(sql_examples)
            self.logger.info(f"[DOCS] SUCCESS Loaded {len(sql_documents)} SQL examples")
            
            # Load documentation
            self.logger.info("[DOCS] Loading documentation...")
            doc_documents = loader.load_documentation()
            self.logger.info(f"[DOCS] SUCCESS Loaded {len(doc_documents)} documentation files")
            
            # Combine documents
            all_documents = sql_documents + doc_documents
            
            if not all_documents:
                self.logger.warning("[DOCS] WARNING No documents found")
                return
            
            # Chunk documents
            self.logger.info("[DOCS] PROCESSING Chunking documents...")
            chunked_docs = loader.chunk_documents(
                all_documents,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            self.logger.info(f"[DOCS] SUCCESS Created {len(chunked_docs)} chunks")
            
            # Add to vector store using fast method
            await self._add_documents_fast(chunked_docs, "knowledge_base", batch_size=10)
            
            # Final save
            self._save_persisted_data()
            
            self.logger.info("[DOCS] SUCCESS Knowledge base loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"[DOCS] ERROR Error loading knowledge base: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store with enhanced processing."""
        await self._add_documents_fast(documents, "user_documents", batch_size=5)
    
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        score_threshold: float = -1.0,  # Threshold más permisivo por defecto
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores and optional filtering.
        Enhanced version with filtering support and improved threshold.
        """
        try:
            if not self.documents or self.embeddings is None:
                self.logger.warning("[SEARCH] WARNING No documents available for search")
                return []
            
            # Ensure embedding model is loaded
            self._ensure_embedding_model()
            
            # Calculate query embedding
            query_embedding = self._embedding_model.encode([query])
            
            # Validate embedding dimensions
            if query_embedding.shape[1] != self.embeddings.shape[1]:
                self.logger.error(f"[SEARCH] ERROR Dimension mismatch: query {query_embedding.shape[1]} vs stored {self.embeddings.shape[1]}")
                return []
            
            # Calculate cosine similarities efficiently
            # Normalize embeddings (L2 normalization)
            embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)
            
            # Cosine similarity = normalized_embeddings · normalized_query
            similarities = np.dot(embeddings_norm, query_norm.T).flatten()
            
            # Ensure similarities are in valid range [-1, 1]
            similarities = np.clip(similarities, -1.0, 1.0)
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k * 2]  # Get more for filtering
            
            # Build results with filtering
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                
                if score >= score_threshold:
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    
                    # Apply filter if provided
                    if filter_dict:
                        match = all(metadata.get(key) == value for key, value in filter_dict.items())
                        if not match:
                            continue
                    
                    document = Document(
                        page_content=self.documents[idx],
                        metadata=metadata
                    )
                    results.append((document, score))
                    
                    # Stop when we have enough results
                    if len(results) >= k:
                        break
            
            self.logger.info(f"[SEARCH] FOUND Found {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            self.logger.error(f"[SEARCH] ERROR Search error: {e}")
            return []
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents (compatibility method)."""
        results = await self.similarity_search_with_score(query, k, filter_dict=filter_dict)
        return [doc for doc, score in results]
    
    def _safe_similarity_search(self, query: str, k: int, filter_dict: dict = None) -> List[Document]:
        """Synchronous similarity search with error handling."""
        try:
            # Check if we're in an async context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in async context - cannot use asyncio.run()
                # Log warning and return empty results
                self.logger.warning(f"[SEARCH] WARNING Cannot run sync search in async context for query: '{query[:50]}...'")
                return []
            else:
                # Safe to run async method in sync context
                return asyncio.run(self.similarity_search(query, k, filter_dict=filter_dict))
        except Exception as e:
            self.logger.error(f"[SEARCH] ERROR Safe search error: {e}")
            return []
    
    async def _get_collection_count(self) -> int:
        """Get count of documents in collection with timeout protection."""
        try:
            return len(self.documents)
        except Exception as e:
            self.logger.warning(f"[COUNT] WARNING Error getting count: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector store statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "embedding_model": self._embedding_model_name,
            "initialized": self._initialized,
            "loading_started": self._loading_started,
            "has_embeddings": self.embeddings is not None,
            "metadata_count": len(self.metadata),
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size
        }
    
    def reset_vector_store(self):
        """Reset the vector store completely."""
        try:
            self.logger.info("[RESET] DELETING Resetting vector store...")
            
            # Clear in-memory data
            self.documents = []
            self.embeddings = None
            self.metadata = []
            self.clear_embedding_cache()
            
            # Remove persisted files
            for file_path in [self.documents_file, self.embeddings_file, self.metadata_file]:
                if file_path.exists():
                    file_path.unlink()
            
            # Reset flags
            self._initialized = False
            self._loading_started = False
            
            self.logger.info("[RESET] SUCCESS Vector store reset successfully")
            
        except Exception as e:
            self.logger.error(f"[RESET] ERROR Error resetting vector store: {e}")
            raise


# For compatibility with existing code
class VectorStore:
    """
    Compatibility wrapper for existing code.
    Enhanced version with better method forwarding.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._corporate_store = None
    
    @property
    def embeddings(self):
        """Dummy property for compatibility."""
        return None
    
    def clear_embedding_cache(self):
        """Clear embedding cache."""
        if self._corporate_store:
            self._corporate_store.clear_embedding_cache()
    
    async def initialize(self) -> None:
        """Initialize the corporate vector store."""
        self.logger.info("[CORPORATE] INITIALIZING Initializing Enhanced Corporate Vector Store...")
        
        self._corporate_store = CorporateVectorStore()
        await self._corporate_store.initialize()
        
        self.logger.info("[CORPORATE] SUCCESS Enhanced Corporate Vector Store ready!")
    
    async def load_knowledge_base(self) -> None:
        """Load knowledge base."""
        if self._corporate_store:
            await self._corporate_store.load_knowledge_base()
    
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search with scores."""
        if self._corporate_store:
            return await self._corporate_store.similarity_search_with_score(
                query, k, score_threshold, filter_dict
            )
        return []
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents."""
        if self._corporate_store:
            return await self._corporate_store.similarity_search(query, k, filter_dict, **kwargs)
        return []
    
    def _safe_similarity_search(self, query: str, k: int, filter_dict: dict = None) -> List[Document]:
        """Safe synchronous search."""
        if self._corporate_store:
            return self._corporate_store._safe_similarity_search(query, k, filter_dict)
        return []
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents."""
        if self._corporate_store:
            await self._corporate_store.add_documents(documents)
    
    async def _get_collection_count(self) -> int:
        """Get collection count."""
        if self._corporate_store:
            return await self._corporate_store._get_collection_count()
        return 0
    
    async def search_similar_examples(self, query: str, k: int = None) -> List[Document]:
        """Search for similar SQL examples - compatibility method."""
        if self._corporate_store:
            from config.settings import settings
            k = k or getattr(settings, 'top_k_retrieval', 4)
            
            # Use similarity search and filter for SQL examples
            results = await self._corporate_store.similarity_search(query, k * 2)  # Get more to filter
            
            # Filter for SQL examples based on metadata
            sql_examples = []
            for doc in results:
                if doc.metadata.get('type') == 'sql_example' or 'SQL' in doc.page_content:
                    sql_examples.append(doc)
                    if len(sql_examples) >= k:
                        break
            
            return sql_examples[:k]
        return []
    
    async def search_documentation(self, query: str, k: int = None) -> List[Document]:
        """Search for relevant documentation - compatibility method.""" 
        if self._corporate_store:
            from config.settings import settings
            k = k or getattr(settings, 'top_k_retrieval', 4)
            
            # Use similarity search and filter for documentation
            results = await self._corporate_store.similarity_search(query, k * 2)
            
            # Filter for documentation based on metadata
            docs = []
            for doc in results:
                if doc.metadata.get('type') == 'documentation' or doc.metadata.get('source', '').endswith('.txt'):
                    docs.append(doc)
                    if len(docs) >= k:
                        break
            
            return docs[:k]
        return []
    
    async def search_ok_examples_by_category(self, query: str, category: str = None, k: int = None) -> List[Document]:
        """Search for OK SQL examples by category - compatibility method."""
        if self._corporate_store:
            from config.settings import settings
            k = k or getattr(settings, 'top_k_retrieval', 4)
            
            # Use similarity search and filter for OK examples
            results = await self._corporate_store.similarity_search(query, k * 3)
            
            # Filter for OK examples and specific category if provided
            ok_examples = []
            for doc in results:
                metadata = doc.metadata
                is_ok = metadata.get('example_type') == 'OK' or metadata.get('label') == 'OK' or 'OK' in doc.page_content
                matches_category = category is None or metadata.get('category') == category
                
                if is_ok and matches_category:
                    ok_examples.append(doc)
                    if len(ok_examples) >= k:
                        break
            
            return ok_examples[:k]
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        if self._corporate_store:
            return self._corporate_store.get_stats()
        return {}
    
    def reset_vector_store(self):
        """Reset vector store."""
        if self._corporate_store:
            self._corporate_store.reset_vector_store()


# Test function for validation
async def test_enhanced_corporate_vector_store():
    """Test the enhanced corporate vector store."""
    
    print("TESTING ENHANCED CORPORATE VECTOR STORE")
    print("=" * 60)
    
    try:
        # Test direct corporate store
        print("\n1. Testing Enhanced CorporateVectorStore...")
        store = CorporateVectorStore()
        await store.initialize()
        
        # Test stats
        stats = store.get_stats()
        print(f"STATS Initial stats: {len(stats)} fields")
        
        # Test compatibility wrapper
        print("\n2. Testing Enhanced VectorStore wrapper...")
        wrapper = VectorStore()
        await wrapper.initialize()
        
        # Test collection count method
        count = await wrapper._get_collection_count()
        print(f"COUNT Collection count: {count}")
        
        wrapper_stats = wrapper.get_stats()
        print(f"WRAPPER Wrapper stats: {wrapper_stats}")
        
        print("\nSUCCESS ALL ENHANCED TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\nFAILED ENHANCED TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_enhanced_corporate_vector_store())
    exit(0 if success else 1)
