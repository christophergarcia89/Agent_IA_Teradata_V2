"""
Document loader for RAG system.
Loads SQL examples and documentation from the knowledge base.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class SQLExample:
    """SQL example with metadata."""
    content: str
    example_type: str  # "OK" or "NOK"
    category: str  # "UPDATE", "CREATE", "SELECT", etc.
    example_number: str
    file_path: str


class DocumentLoader:
    """Loads and processes documents from the knowledge base."""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.logger = logging.getLogger(__name__)
        
    def load_sql_examples(self) -> List[SQLExample]:
        """Load all SQL examples from consolidated text files with tags."""
        examples = []
        examples_path = self.knowledge_base_path / "examples"
        
        if not examples_path.exists():
            self.logger.warning(f"Examples directory not found: {examples_path}")
            return examples
        
        # Buscar archivos .txt consolidados con timeout protection
        consolidated_files = list(examples_path.glob("*.txt"))
        self.logger.info(f"Found {len(consolidated_files)} consolidated files to process")
        
        for file_path in consolidated_files:
            try:
                self.logger.info(f"Processing consolidated file: {file_path.name}")
                consolidated_examples = self._parse_consolidated_file(file_path)
                examples.extend(consolidated_examples)
                self.logger.info(f"Loaded {len(consolidated_examples)} examples from {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error loading consolidated file {file_path}: {e}")
        
        # También mantener compatibilidad con archivos .sql individuales
        sql_files = list(examples_path.glob("*.sql"))
        if sql_files:
            self.logger.info(f"Found {len(sql_files)} individual SQL files to process")
            
            for file_path in sql_files:
                try:
                    example = self._parse_sql_file(file_path)
                    if example:
                        examples.append(example)
                except Exception as e:
                    self.logger.error(f"Error loading SQL file {file_path}: {e}")
        
        self.logger.info(f"[SUCCESS] Total loaded: {len(examples)} SQL examples")
        
        # Log statistics for debugging
        stats = self.get_example_statistics(examples)
        self.logger.info(f"[STATS] Examples by type - OK: {stats['ok_examples']}, NOK: {stats['nok_examples']}")
        self.logger.info(f"[CATEGORIES] Categories found: {list(stats['categories'].keys())}")
        
        return examples
    
    def load_documentation(self) -> List[Document]:
        """Load documentation files with enhanced metadata and chunking."""
        documents = []
        docs_path = self.knowledge_base_path / "documentation"
        
        if not docs_path.exists():
            self.logger.warning(f"Documentation directory not found: {docs_path}")
            return documents
            
        for file_path in docs_path.glob("*.txt"):
            try:
                self.logger.info(f"Loading documentation file: {file_path.name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Metadata enriquecida
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "type": "documentation",
                    "doc_type": "documentation",
                    "source_type": "documentation",
                    "document_type": "documentation",
                    "category": "documentation",
                    "file_type": "txt",
                    "path": str(file_path.parent)
                }

                # Dividir el contenido en chunks para mejor procesamiento
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", ", ", " ", ""]
                )
                
                chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))

            except Exception as e:
                self.logger.error(f"Error loading documentation file {file_path}: {e}")
                continue

        self.logger.info(f"Loaded {len(documents)} documentation chunks")
        return documents
    
    def _parse_consolidated_file(self, file_path: Path) -> List[SQLExample]:
        """Parse consolidated text file with multiple examples separated by tags."""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir contenido por etiquetas que terminan en .sql
            # Patrón: busca líneas que terminen en "NOK.sql" o "OK.sql"
            import re
            
            # Buscar todas las etiquetas y sus posiciones
            tag_pattern = r'^(.+?\s+-\s+Ejemplo\s+\d+\s+(?:OK|NOK)\.sql)\s*$'
            tags = []
            
            for match in re.finditer(tag_pattern, content, re.MULTILINE):
                tag_text = match.group(1).strip()
                start_pos = match.end()
                tags.append((tag_text, start_pos))
            
            # Procesar cada ejemplo
            for i, (tag_text, start_pos) in enumerate(tags):
                # Determinar el final del ejemplo (inicio del siguiente o final del archivo)
                if i + 1 < len(tags):
                    end_pos = tags[i + 1][1]
                    # Buscar hacia atrás desde el inicio del próximo tag para encontrar el final real
                    next_tag_start = content.rfind('\n', 0, tags[i + 1][1] - len(tags[i + 1][0]))
                    end_pos = next_tag_start if next_tag_start > start_pos else end_pos
                else:
                    end_pos = len(content)
                
                # Extraer el contenido del ejemplo
                example_content = content[start_pos:end_pos].strip()
                
                # Limpiar contenido (remover líneas vacías al inicio y final)
                example_content = example_content.strip()
                
                if not example_content:
                    continue
                
                # Parsear información de la etiqueta
                example_info = self._parse_tag_info(tag_text)
                if not example_info:
                    self.logger.warning(f"Could not parse tag: {tag_text}")
                    continue
                
                # Crear SQLExample
                sql_example = SQLExample(
                    content=example_content,
                    example_type=example_info['type'],
                    category=example_info['category'],
                    example_number=example_info['number'],
                    file_path=str(file_path)
                )
                
                examples.append(sql_example)
                
                self.logger.debug(f"Parsed example: {example_info['category']} - Ejemplo {example_info['number']} {example_info['type']}")
        
        except Exception as e:
            self.logger.error(f"Error parsing consolidated file {file_path}: {e}")
        
        return examples
    
    def _parse_tag_info(self, tag_text: str) -> dict:
        """Parse tag information to extract category, number, and type."""
        # Patrón esperado: "CATEGORY - Ejemplo X TYPE.sql"
        # Ejemplos: "CREATE_UPDATE - Ejemplo 1 NOK.sql", "SELECT - Ejemplo 2 OK.sql"
        
        import re
        
        # Patrón más flexible para manejar diferentes formatos
        pattern = r'^(.+?)\s+-\s+Ejemplo\s+(\d+[A-Z]*)\s+(OK|NOK)\.sql'
        match = re.match(pattern, tag_text.strip(), re.IGNORECASE)
        
        if match:
            category = match.group(1).strip()
            number = match.group(2).strip()
            example_type = match.group(3).upper()
            
            return {
                'category': category,
                'number': number,
                'type': example_type
            }
        
        # Intentar patrón alternativo sin "Ejemplo"
        alt_pattern = r'^(.+?)\s+-\s+(\d+[A-Z]*)\s+(OK|NOK)\.sql'
        alt_match = re.match(alt_pattern, tag_text.strip(), re.IGNORECASE)
        
        if alt_match:
            category = alt_match.group(1).strip()
            number = alt_match.group(2).strip()
            example_type = alt_match.group(3).upper()
            
            return {
                'category': category,
                'number': number,
                'type': example_type
            }
        
        return None
    
    def _parse_sql_file(self, file_path: Path) -> SQLExample:
        filename = file_path.stem
        
        # Parse filename pattern: "CATEGORY - Ejemplo X TYPE.sql"
        # Examples: "UPDATE - Ejemplo 1 OK.sql", "CREATE - Ejemplo 2 NOK.sql"
        parts = filename.split(" - ")
        if len(parts) < 2:
            self.logger.warning(f"Unexpected filename format: {filename}")
            return None
            
        category = parts[0].strip()
        example_part = parts[1].strip()
        
        # Extract example number and type
        if "NOK" in example_part:
            example_type = "NOK"
            example_number = example_part.replace("NOK", "").replace("Ejemplo", "").strip()
        elif "OK" in example_part:
            example_type = "OK"
            example_number = example_part.replace("OK", "").replace("Ejemplo", "").strip()
        else:
            self.logger.warning(f"Could not determine type (OK/NOK) for: {filename}")
            return None
            
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
            
        return SQLExample(
            content=content,
            example_type=example_type,
            category=category,
            example_number=example_number,
            file_path=str(file_path)
        )
    
    def create_documents_from_examples(self, examples: List[SQLExample]) -> List[Document]:
        """Convert SQL examples to LangChain documents."""
        documents = []
        
        for example in examples:
            doc = Document(
                page_content=example.content,
                metadata={
                    "source": example.file_path,
                    "type": "sql_example",
                    "example_type": example.example_type,
                    "category": example.category,
                    "example_number": example.example_number,
                    "filename": Path(example.file_path).name
                }
            )
            documents.append(doc)
            
        return documents
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Document]:
        """Split documents into chunks for better retrieval with optimized settings."""
        
        # Optimize chunk settings based on document count
        if len(documents) > 100:
            # For large document sets, use smaller chunks for faster processing
            chunk_size = min(chunk_size, 800)
            chunk_overlap = min(chunk_overlap, 150)
            self.logger.info(f"Using optimized chunking for large document set: chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Optimize for SQL content
            separators=["\n\n", "\n", ";", "--", "/*", "*/", " ", ""],
        )
        
        self.logger.info(f"Starting chunking of {len(documents)} documents...")
        chunked_docs = text_splitter.split_documents(documents)
        
        self.logger.info(f"[SUCCESS] Split {len(documents)} documents into {len(chunked_docs)} chunks")
        
        # Log chunking statistics
        if chunked_docs:
            avg_chunk_size = sum(len(doc.page_content) for doc in chunked_docs) / len(chunked_docs)
            self.logger.debug(f"[STATS] Average chunk size: {avg_chunk_size:.0f} characters")
        
        return chunked_docs
    
    def organize_examples_by_pairs(self, examples: List[SQLExample]) -> Dict[str, Tuple[SQLExample, SQLExample]]:
        """Organize examples into OK/NOK pairs for comparison."""
        pairs = {}
        
        # Group by category and example number
        grouped = {}
        for example in examples:
            # Crear clave que maneje diferentes formatos de número (1, 1A, 1B, etc.)
            key = f"{example.category}_{example.example_number}"
            if key not in grouped:
                grouped[key] = {}
            grouped[key][example.example_type] = example
            
        # Create pairs
        for key, group in grouped.items():
            if "OK" in group and "NOK" in group:
                pairs[key] = (group["OK"], group["NOK"])
            else:
                self.logger.warning(f"Incomplete pair for {key}: {list(group.keys())}")
                
        self.logger.info(f"Created {len(pairs)} OK/NOK pairs")
        return pairs
    
    def get_examples_by_category(self, examples: List[SQLExample], category: str) -> Dict[str, List[SQLExample]]:
        """Get examples grouped by type for a specific category."""
        category_examples = {"OK": [], "NOK": []}
        
        for example in examples:
            if example.category.upper() == category.upper():
                if example.example_type in category_examples:
                    category_examples[example.example_type].append(example)
        
        return category_examples
    
    def get_example_statistics(self, examples: List[SQLExample]) -> Dict[str, Any]:
        """Get statistics about loaded examples."""
        stats = {
            "total_examples": len(examples),
            "ok_examples": 0,
            "nok_examples": 0,
            "categories": {},
            "files_processed": set()
        }
        
        for example in examples:
            # Count by type
            if example.example_type == "OK":
                stats["ok_examples"] += 1
            elif example.example_type == "NOK":
                stats["nok_examples"] += 1
            
            # Count by category
            category = example.category
            if category not in stats["categories"]:
                stats["categories"][category] = {"OK": 0, "NOK": 0}
            
            if example.example_type in ["OK", "NOK"]:
                stats["categories"][category][example.example_type] += 1
            
            # Track files
            stats["files_processed"].add(Path(example.file_path).name)
        
        # Convert set to list for JSON serialization
        stats["files_processed"] = list(stats["files_processed"])
        
        return stats