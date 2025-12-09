"""
Text chunking strategies for RAG systems.

Different strategies for breaking documents into optimal-sized chunks:
- Fixed-size: Simple token-based splitting
- Sentence-aware: Respects sentence boundaries
- Semantic: Groups semantically related content
- Recursive: Hierarchical splitting
"""

import re
from typing import List, Dict, Optional
from pathlib import Path

import tiktoken
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    NLTKTextSplitter
)

from utils.config import config
from utils.logger import logger


class Chunker:
    """Main chunking interface with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: Chunking strategy ('fixed_size', 'sentence_aware', 'recursive', 'semantic')
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.strategy = strategy or config.chunk_strategy
        
        # Initialize tokenizer for counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Failed to load tiktoken, using approximate token counting")
            self.tokenizer = None
        
        logger.info(f"Initialized chunker: strategy={self.strategy}, size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if self.strategy == "fixed_size":
            return self._chunk_fixed_size(text, metadata)
        elif self.strategy == "sentence_aware":
            return self._chunk_sentence_aware(text, metadata)
        elif self.strategy == "recursive":
            return self._chunk_recursive(text, metadata)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, metadata)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using fixed_size")
            return self._chunk_fixed_size(text, metadata)
    
    def _chunk_fixed_size(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Fixed-size chunking: Simple token-based splitting.
        
        Pros: Fast, predictable chunk sizes
        Cons: May split sentences/paragraphs mid-way
        """
        chunks = []
        
        if self.tokenizer:
            # Tokenize and split by tokens
            tokens = self.tokenizer.encode(text)
            overlap_tokens = self._approximate_overlap_tokens()
            
            for i in range(0, len(tokens), self.chunk_size - overlap_tokens):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "strategy": "fixed_size",
                    "token_count": len(chunk_tokens)
                }
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        else:
            # Fallback: approximate 4 chars per token
            char_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4
            
            for i in range(0, len(text), char_size - char_overlap):
                chunk_text = text[i:i + char_size]
                
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "strategy": "fixed_size",
                    "approximate_tokens": len(chunk_text) // 4
                }
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def _chunk_sentence_aware(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Sentence-aware chunking: Splits at sentence boundaries.
        
        Pros: Preserves sentence integrity
        Cons: Chunk sizes may vary significantly
        """
        try:
            # Use LangChain's recursive splitter with sentence separators
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=self._count_tokens
            )
            
            chunks_text = splitter.split_text(text)
            
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": i,
                    "strategy": "sentence_aware",
                    "token_count": self._count_tokens(chunk_text)
                }
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunks)} sentence-aware chunks")
            return chunks
            
        except Exception as e:
            logger.warning(f"Sentence-aware chunking failed: {e}, falling back to fixed_size")
            return self._chunk_fixed_size(text, metadata)
    
    def _chunk_recursive(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Recursive chunking: Hierarchical splitting (paragraph -> sentence -> word).
        
        Pros: Tries to keep related content together
        Cons: Can be slower
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",      # Paragraphs
                    "\n",        # Lines
                    ". ",        # Sentences
                    "! ", "? ",  # More sentence endings
                    " ",         # Words
                    ""           # Characters (last resort)
                ],
                length_function=self._count_tokens
            )
            
            chunks_text = splitter.split_text(text)
            
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": i,
                    "strategy": "recursive",
                    "token_count": self._count_tokens(chunk_text)
                }
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunks)} recursive chunks")
            return chunks
            
        except Exception as e:
            logger.warning(f"Recursive chunking failed: {e}, falling back to fixed_size")
            return self._chunk_fixed_size(text, metadata)
    
    def _chunk_semantic(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Semantic chunking: Groups semantically related sentences.
        
        Note: This is a simplified version. Full semantic chunking requires
        embeddings and similarity calculations, which is more complex.
        
        Pros: Groups related content, better retrieval quality
        Cons: Slower, requires embedding model
        """
        try:
            # For now, use sentence-based splitting with better boundaries
            # Full semantic chunking would require embeddings and similarity
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self._count_tokens(sentence)
                
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_metadata = {
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                        "strategy": "semantic",
                        "token_count": current_tokens
                    }
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
                    
                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap_sentences(current_chunk)
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(self._count_tokens(s) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            # Add last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "strategy": "semantic",
                    "token_count": current_tokens
                }
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}, falling back to fixed_size")
            return self._chunk_fixed_size(text, metadata)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: ~4 characters per token
            return len(text) // 4
    
    def _approximate_overlap_tokens(self) -> int:
        """Calculate overlap in tokens."""
        return min(self.chunk_overlap, self.chunk_size // 4)
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get last N sentences for overlap."""
        overlap_count = max(1, len(sentences) // 4)
        return sentences[-overlap_count:]


def chunk_document(content: Dict[str, any], chunker: Optional[Chunker] = None) -> List[Dict[str, any]]:
    """
    Chunk a document (from scraper).
    
    Args:
        content: Document content dictionary
        chunker: Chunker instance (creates new if None)
        
    Returns:
        List of chunked documents
    """
    if chunker is None:
        chunker = Chunker()
    
    metadata = {
        "source_url": content.get("url"),
        "source_title": content.get("title"),
        "source_date": content.get("date"),
        "scraped_at": content.get("scraped_at")
    }
    
    chunks = chunker.chunk_text(content["content"], metadata)
    
    return chunks
