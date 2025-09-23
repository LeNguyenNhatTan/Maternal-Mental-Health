import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
import requests
import hashlib
from datetime import datetime

from .document_processor import Document, Chunker

class SimpleVectorStore:
    """Enhanced vector store with persistent storage and better caching."""
    
    CACHE_VERSION = "2.0"  # Change this to force cache refresh
    METADATA_FILE = "vectorstore_metadata.json"
    
    def __init__(self, embedding_dim: int = 768, cache_dir: str = None, refresh_cache: bool = False):
        """
        Initialize the vector store with persistent storage.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            cache_dir: Directory to cache embeddings
            refresh_cache: Force refresh of cache (ignore existing cache)
        """
        self.documents = []
        self.embeddings = []
        self.embedding_dim = embedding_dim
        self.refresh_cache = refresh_cache
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / self.METADATA_FILE
        
        # Try to load existing cache first
        if not self.refresh_cache and self._load_from_cache():
            print(f"Loaded {len(self.documents)} cached documents and embeddings")
        else:
            print("Cache refresh requested or cache not found - will rebuild")
            if self.refresh_cache:
                self._clear_cache()
    
    def _load_from_cache(self) -> bool:
        """Load vector store from persistent cache."""
        try:
            # Check if metadata file exists and is compatible
            if not self.metadata_file.exists():
                return False
            
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check cache version compatibility
            if metadata.get('version', '1.0') != self.CACHE_VERSION:
                print(f"Cache version mismatch: {metadata.get('version')} vs {self.CACHE_VERSION}")
                return False
            
            # Check if cache files exist
            cache_timestamp = metadata.get('timestamp', 0)
            if cache_timestamp == 0:
                return False
            
            # Load documents and embeddings
            documents_file = self.cache_dir / f"documents_{cache_timestamp}.pkl"
            embeddings_file = self.cache_dir / f"embeddings_{cache_timestamp}.pkl"
            
            if not (documents_file.exists() and embeddings_file.exists()):
                print("Cache files missing - will rebuild")
                return False
            
            # Load data
            with open(documents_file, 'rb') as f:
                self.documents = pickle.load(f)
            
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            
            # Verify data integrity
            if len(self.documents) != len(self.embeddings):
                print("Cache corruption detected - document/embedding count mismatch")
                return False
            
            # Verify embedding dimensions
            if self.embeddings and len(self.embeddings[0]) != self.embedding_dim:
                print(f"Embedding dimension mismatch: {len(self.embeddings[0])} vs {self.embedding_dim}")
                return False
            
            print(f"Successfully loaded cache from {datetime.fromtimestamp(cache_timestamp)}")
            return True
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """Save vector store to persistent cache."""
        try:
            # Create timestamp for this cache
            cache_timestamp = int(time.time())
            
            # Save documents
            documents_file = self.cache_dir / f"documents_{cache_timestamp}.pkl"
            with open(documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save embeddings
            embeddings_file = self.cache_dir / f"embeddings_{cache_timestamp}.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            metadata = {
                'version': self.CACHE_VERSION,
                'embedding_dim': self.embedding_dim,
                'num_documents': len(self.documents),
                'num_embeddings': len(self.embeddings),
                'timestamp': cache_timestamp,
                'created': datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Clean up old cache files (keep last 3 versions)
            self._cleanup_old_cache(cache_timestamp)
            
            print(f"💾 Saved cache: {len(self.documents)} documents, {cache_timestamp}")
            
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _cleanup_old_cache(self, current_timestamp: int) -> None:
        """Clean up old cache files, keeping only the last 3 versions."""
        try:
            # Find all cache files with timestamps
            pattern = self.cache_dir / f"{'documents','embeddings'}_{'*'}.pkl"
            all_files = list(self.cache_dir.glob("documents_*.pkl")) + list(self.cache_dir.glob("embeddings_*.pkl"))
            
            if len(all_files) <= 6:  # 3 versions × 2 files each
                return
            
            # Extract timestamps and sort (oldest first)
            timestamped_files = []
            for file in all_files:
                try:
                    # Extract timestamp from filename
                    filename = file.name
                    if filename.startswith(('documents_', 'embeddings_')) and filename.endswith('.pkl'):
                        timestamp_str = filename.split('_')[1].replace('.pkl', '')
                        timestamp = int(timestamp_str)
                        timestamped_files.append((timestamp, file))
                except (ValueError, IndexError):
                    continue
            
            timestamped_files.sort(key=lambda x: x[0])  # Sort by timestamp ascending
            
            # Delete oldest files (keep last 3 versions = 6 files)
            files_to_delete = timestamped_files[:-6]  # All except last 6
            
            for _, file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"Deleted old cache file: {file_path.name}")
                except Exception as e:
                    print(f"Could not delete {file_path.name}: {e}")
                    
        except Exception as e:
            print(f"⚠️  Error during cache cleanup: {e}")
    
    def _clear_cache(self) -> None:
        """Clear all cache files."""
        try:
            for file in self.cache_dir.glob("*.pkl"):
                if file.name.startswith(('documents_', 'embeddings_')):
                    file.unlink()
                    print(f"Deleted: {file.name}")
            
            if self.metadata_file.exists():
                self.metadata_file.unlink()
                print("Deleted metadata file")
                
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def add_documents(self, documents: List[Document], chunk_size: int = 1000) -> None:
        """
        Add documents to the vector store with improved caching.
        
        Args:
            documents: List of documents to add
            chunk_size: Size of chunks to split documents into
        """
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Chunk documents
        chunker = Chunker()
        all_chunks = []
        
        for doc in documents:
            chunks = chunker.chunk_document(doc, chunk_size=chunk_size)
            all_chunks.extend(chunks)
            print(f"{doc.metadata.get('filename', 'Unknown')}: {len(chunks)} chunks")
        
        print(f"Created {len(all_chunks)} total chunks")
        
        # Compute embeddings for chunks with progress tracking
        new_embeddings_count = 0
        cache_hits = 0
        
        for i, chunk in enumerate(all_chunks):
            if i % 50 == 0 and i > 0:
                print(f"   Progress: {i}/{len(all_chunks)} chunks processed")
            
            embedding = self._get_embedding(chunk.content)
            
            if embedding is not None:
                self.documents.append(chunk)
                self.embeddings.append(embedding)
                
                if hasattr(self, '_last_cache_check') and self._last_cache_check:
                    cache_hits += 1
                new_embeddings_count += 1
        
        print(f"Added {len(all_chunks)} chunks: {cache_hits} cache hits, {new_embeddings_count} new embeddings")
        
        # Save to persistent cache after processing
        if new_embeddings_count > 0:
            self._save_to_cache()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.documents:
            print("No documents in vector store for search")
            return []
        
        # Get query embedding with caching
        query_embedding = self._get_embedding(query)
        
        if query_embedding is None:
            print("Could not generate query embedding")
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, similarity in similarities[:top_k]:
            results.append((self.documents[i], similarity))
        
        return results
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text using Ollama API with enhanced caching.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector or None if embedding failed
        """
        if not text.strip():
            return None
        
        # Create stable cache key (hash of text content)
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"embedding_{text_hash}.pkl"
        
        # Check cache first
        if cache_file.exists() and not self.refresh_cache:
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Verify embedding dimensions
                if len(embedding) == self.embedding_dim:
                    self._last_cache_check = True
                    return embedding
                else:
                    print(f"Cache dimension mismatch, recomputing: {len(embedding)} vs {self.embedding_dim}")
                    cache_file.unlink()
            except Exception as e:
                print(f"Cache read error: {e}")
                if cache_file.exists():
                    cache_file.unlink()
        
        self._last_cache_check = False
        
        # If no cache or refresh requested, compute embedding
        try:
            embedding_model = "nomic-embed-text:latest"
            
            # Check available models
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    available_models = [model["name"] for model in response.json()["models"]]
                    
                    if embedding_model not in available_models:
                        print(f"'{embedding_model}' not found. Available: {', '.join([m for m in available_models if 'embed' in m.lower()] or ['none'])}")
                        
                        # Try alternative embedding models
                        for model in available_models:
                            if "embed" in model.lower():
                                embedding_model = model
                                print(f"   Using: {embedding_model}")
                                break
                        else:
                            # Fallback to first available model
                            if available_models:
                                embedding_model = available_models[0]
                                print(f"   Fallback to: {embedding_model}")
                            else:
                                raise ValueError("No models available in Ollama")
            except Exception as e:
                print(f"Error checking models: {e}. Using default: {embedding_model}")
            
            # Generate embedding
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": embedding_model, "prompt": text},
                timeout=30
            )
            
            if response.status_code == 200:
                embedding_data = response.json()
                embedding = np.array(embedding_data["embedding"])
                
                # Verify dimensions
                if len(embedding) != self.embedding_dim:
                    print(f"Unexpected embedding dimension: {len(embedding)} (expected: {self.embedding_dim})")
                    return None
                
                # Cache the embedding
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(embedding, f)
                except Exception as e:
                    print(f"Cache write error: {e}")
                
                print(f"New embedding cached: {text[:50]}...")
                return embedding
            else:
                print(f"Embedding API error {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print("Embedding request timed out")
        except Exception as e:
            print(f"Embedding error: {e}")
        
        # Fallback: Generate random normalized embedding
        print("Using fallback random embedding")
        fallback = np.random.rand(self.embedding_dim).astype(np.float32)
        fallback = fallback / np.linalg.norm(fallback)
        return fallback
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def save(self, file_path: str) -> None:
        """Save the vector store to a specific file (legacy method)."""
        print(f"Saving vector store to {file_path}")
        data = {
            'documents': self.documents,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'embedding_dim': self.embedding_dim,
            'version': self.CACHE_VERSION,
            'timestamp': int(time.time())
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'SimpleVectorStore':
        """Load a vector store from a specific file (legacy method)."""
        print(f"Loading vector store from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        store = cls(embedding_dim=data['embedding_dim'])
        store.documents = data['documents']
        store.embeddings = [np.array(emb) for emb in data['embeddings']]
        
        # Auto-save to new cache format
        store._save_to_cache()
        return store
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the current cache."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                return {
                    'version': metadata.get('version', 'unknown'),
                    'num_documents': metadata.get('num_documents', 0),
                    'num_embeddings': metadata.get('num_embeddings', 0),
                    'timestamp': metadata.get('timestamp', 0),
                    'cache_dir': str(self.cache_dir),
                    'cache_file_count': len(list(self.cache_dir.glob("*.pkl")))
                }
            else:
                return {
                    'status': 'no_cache',
                    'cache_dir': str(self.cache_dir),
                    'cache_file_count': len(list(self.cache_dir.glob("*.pkl")))
                }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_cache(self) -> None:
        """Clear the entire cache."""
        print("Clearing vector store cache")
        self._clear_cache()
        self.documents = []
        self.embeddings = []