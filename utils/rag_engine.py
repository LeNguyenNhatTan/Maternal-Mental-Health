import os
import json
import time
import threading
import signal
from typing import List, Dict, Any, Optional, Tuple
from .document_processor import Document, DocumentProcessor, process_documents_directory, extract_questions_from_text
from .vector_store import SimpleVectorStore

# Define a timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

class RAGEngine:
    """Retrieval-Augmented Generation engine for the mental health assistant."""
    
    def __init__(self, documents_dir: str, questionnaire_dir: str = None, cache_dir: str = None, refresh_cache: bool = False):
        """
        Initialize the RAG engine.

        Args:
            documents_dir: Directory containing reference documents
            questionnaire_dir: Directory containing questionnaires (if None, uses documents_dir)
            cache_dir: Directory to cache embeddings
            refresh_cache: Whether to refresh the document cache
        """
        print(f"RAGEngine: Initializing with documents_dir={documents_dir}, refresh_cache={refresh_cache}")
        self.documents_dir = documents_dir
        self.questionnaire_dir = questionnaire_dir or documents_dir
        self.refresh_cache = refresh_cache
        
        # Set default cache directory if not provided
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        
        # Create vector store
        print(f"RAGEngine: Creating vector store with refresh_cache={refresh_cache}")
        self.vector_store = SimpleVectorStore(
            cache_dir=cache_dir, 
            refresh_cache=refresh_cache
        )
        
        # Print cache status
        cache_stats = self.vector_store.get_cache_stats()
        print(f"Cache status: {cache_stats.get('status', 'unknown')}, {cache_stats.get('num_documents', 0)} docs")
        
        # Load and process all documents for RAG
        print("RAGEngine: Loading documents")
        self.document_map = self._load_documents()
        
        # Load questionnaires separately
        print("RAGEngine: Loading questionnaires")
        self.questionnaire_map = self._load_questionnaires()
        
        self.accessed_documents = []  # Track accessed documents
        
        # Semantic similarity model for measuring RAG impact
        self.similarity_model = None
        self._initialize_similarity_model()
        
        # Track RAG metrics
        self.retrieval_stats = {
            "total_retrievals": 0,
            "avg_relevance_score": 0,
            "top_accessed_documents": {}
        }
        print("RAGEngine: Initialization complete")
    
    def _initialize_similarity_model(self):
        """Initialize the similarity model with a timeout mechanism."""
        # Skip SentenceTransformer initialization if in SKIP_TRANSFORMERS environment mode
        if os.environ.get('SKIP_TRANSFORMERS', '').lower() in ('1', 'true', 'yes'):
            print("RAGEngine: Skipping SentenceTransformer initialization due to environment setting")
            return
            
        print("RAGEngine: Attempting to load SentenceTransformer")
        
        # Try importing the module first to check availability
        try:
            import importlib
            if not importlib.util.find_spec("sentence_transformers"):
                print("RAGEngine: SentenceTransformer package not found. Impact measurement disabled.")
                return
                
            print("RAGEngine: SentenceTransformer package found. Will attempt to load model.")
        except ImportError:
            print("RAGEngine: ImportLib not available. Skipping SentenceTransformer check.")
            return
        
        # Define a function that loads the model in a separate thread
        def load_model():
            try:
                from sentence_transformers import SentenceTransformer
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("RAGEngine: SentenceTransformer model loaded successfully")
            except Exception as e:
                print(f"RAGEngine: Error loading SentenceTransformer model: {e}")
                self.similarity_model = None
        
        # Start model loading in a separate thread with a timeout
        model_thread = threading.Thread(target=load_model)
        model_thread.daemon = True  # Allow the thread to be killed when main thread exits
        
        print("RAGEngine: Starting model load in separate thread")
        model_thread.start()
        
        # Wait for at most 30 seconds
        model_thread.join(timeout=30)
        
        if model_thread.is_alive():
            print("RAGEngine: SentenceTransformer model loading timed out. Impact measurement disabled.")
            # The thread is still running but we won't wait for it
    
    def _load_documents(self) -> Dict[str, List[Document]]:
        """Load reference documents from the documents directory."""
        print(f"Loading reference documents from {self.documents_dir}")
        document_map = process_documents_directory(self.documents_dir)
        
        # Add all documents to vector store
        all_docs = []
        for docs in document_map.values():
            all_docs.extend(docs)
        
        self.vector_store.add_documents(all_docs)
        
        return document_map
    
    def _load_questionnaires(self) -> Dict[str, List[str]]:
        """Load questionnaires from the questionnaire directory."""
        questionnaire_map = {}
        
        # Default EPDS questions as fallback
        default_epds_questions = [
            "I have been able to laugh and see the funny side of things",
            "I have looked forward with enjoyment to things",
            "I have blamed myself unnecessarily when things went wrong",
            "I have been anxious or worried for no good reason",
            "I have felt scared or panicky for no very good reason",
            "Things have been getting on top of me",
            "I have been so unhappy that I have had difficulty sleeping",
            "I have felt sad or miserable",
            "I have been so unhappy that I have been crying",
            "The thought of harming myself has occurred to me"
        ]
        
        print(f"Loading questionnaires from {self.questionnaire_dir}")
        
        if self.questionnaire_dir == self.documents_dir:
            # If using the same directory, we've already processed these documents
            if '.pdf' in self.document_map:
                for document in self.document_map['.pdf']:
                    questions = extract_questions_from_text(document.content)
                    filename = document.metadata.get('filename', 'Unknown')
                    print(f"  - {filename}: Found {len(questions)} questions")
                    if questions:
                        questionnaire_map[filename] = questions
                    else:
                        print(f"  - {filename}: No questions found, using default EPDS questions")
                        questionnaire_map[filename] = default_epds_questions
        else:
            # Process the questionnaire directory separately
            questionnaire_docs = process_documents_directory(self.questionnaire_dir)
            
            # Extract questions from all document types
            for ext, docs in questionnaire_docs.items():
                for document in docs:
                    questions = extract_questions_from_text(document.content)
                    filename = document.metadata.get('filename', 'Unknown')
                    print(f"  - {filename}: Found {len(questions)} questions")
                    if questions:
                        questionnaire_map[filename] = questions
                    else:
                        print(f"  - {filename}: No questions found, using default EPDS questions")
                        questionnaire_map[filename] = default_epds_questions
        
        # If no questionnaires found, use default EPDS questions
        if not questionnaire_map:
            print("RAGEngine: No questionnaires detected. Using default EPDS questions.")
            questionnaire_map["default_epds"] = default_epds_questions
        
        return questionnaire_map
    
    def get_questionnaires(self) -> Dict[str, List[str]]:
        """
        Get available questionnaires.
        
        Returns:
            Dictionary mapping questionnaire names to lists of questions
        """
        return self.questionnaire_map
    
    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.0) -> Dict[str, Any]:
        """
        Retrieve relevant documents based on a query with enhanced metadata.
        
        Args:
            query: Query string
            top_k: Number of results to return
            threshold: Minimum relevance score threshold (0.0-1.0)
            
        Returns:
            Dictionary containing:
                - content_list: List of document contents
                - documents: List of document metadata including relevance scores
                - stats: Retrieval statistics
        """
        self.accessed_documents = []  # Reset accessed documents for this query
        self.retrieval_stats["total_retrievals"] += 1
        
        # Request more documents to account for potential duplicates
        search_k = top_k * 4
        
        # Get results with scores
        results = self.vector_store.search(query, top_k=search_k)
        
        # Filter by threshold and prepare return data
        filtered_results = []
        content_list = []
        documents = []
        total_score = 0
        
        # Track documents seen in this query to prevent duplicates
        seen_in_query = set()
        
        for doc, score in results:
            # Stop once we have enough documents
            if len(filtered_results) >= top_k:
                break
                
            if score >= threshold:
                # Create a unique identifier using filename and a hash of content
                source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
                content_hash = hash(doc.content[:100])  # Use first 100 chars for uniqueness
                doc_id = f"{source}|{content_hash}"
                
                # Skip if we've already seen this document in this query
                if doc_id in seen_in_query:
                    # print(f"RAGEngine: Skipping duplicate document in query: {source} (hash: {content_hash})")
                    continue
                    
                # Mark as seen for this query
                seen_in_query.add(doc_id)
                
                filtered_results.append((doc, score))
                content_list.append(doc.content)
                
                # Extract and highlight the most relevant portion of the document
                highlighted_content = self._extract_highlight(doc.content, query)
                
                # Generate a relevance explanation
                relevance_explanation = self._generate_relevance_explanation(
                    query=query, 
                    document_content=doc.content, 
                    score=score, 
                    highlighted_excerpt=highlighted_content
                )
                
                # Prepare document data
                doc_info = {
                    "title": source,
                    "score": round(score, 4),
                    "highlight": highlighted_content,
                    "excerpt": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                    "relevance_explanation": relevance_explanation
                }
                documents.append(doc_info)
                self.accessed_documents.append(doc_info)
                
                # Update document access count
                if source in self.retrieval_stats["top_accessed_documents"]:
                    self.retrieval_stats["top_accessed_documents"][source] += 1
                else:
                    self.retrieval_stats["top_accessed_documents"][source] = 1
                
                total_score += score
        
        # Log if we have fewer documents than requested
        if len(filtered_results) < top_k:
            print(f"RAGEngine: Found only {len(filtered_results)} unique documents after filtering duplicates")
            
        # Calculate average relevance score
        avg_score = total_score / len(filtered_results) if filtered_results else 0
        self.retrieval_stats["avg_relevance_score"] = (
            (self.retrieval_stats["avg_relevance_score"] * (self.retrieval_stats["total_retrievals"] - 1) + avg_score) / 
            self.retrieval_stats["total_retrievals"]
        )
        
        return {
            "content_list": content_list,
            "documents": documents,
            "stats": {
                "query": query,
                "results_count": len(filtered_results),
                "requested_count": top_k,
                "avg_score": round(avg_score, 4)
            }
        }
    
    def _extract_highlight(self, content: str, query: str, context_size: int = 150) -> str:
        """Extract and highlight the most relevant portion of text."""
        # Simple approach: find sentences containing query terms
        query_terms = set(query.lower().split())
        sentences = content.replace("\n", " ").split(". ")
        
        best_sentence = ""
        max_matches = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            matches = sum(1 for term in query_terms if term in sentence.lower())
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence + ("." if not best_sentence.endswith(".") else "")
        
        # Fallback to first part of content
        return content[:context_size] + "..."
    
    def _generate_relevance_explanation(self, query: str, document_content: str, 
                                        score: float, highlighted_excerpt: str) -> str:
        """
        Generate an explanation of why a document is relevant to the query.
        
        Args:
            query: The original query
            document_content: The document content
            score: The relevance score
            highlighted_excerpt: The highlighted excerpt from the document
            
        Returns:
            A brief explanation of document relevance
        """
        # Extract key terms from the query
        query_terms = set(self._extract_key_terms(query))
        
        # Extract key terms from the highlighted excerpt
        excerpt_terms = set(self._extract_key_terms(highlighted_excerpt))
        
        # Find matching terms
        matching_terms = query_terms.intersection(excerpt_terms)
        
        if matching_terms:
            # We found matching terms
            term_list = ", ".join(f'"{term}"' for term in list(matching_terms)[:3])
            if len(matching_terms) > 3:
                term_list += f", and {len(matching_terms) - 3} more"
            
            if score > 0.8:
                return f"High relevance: Contains key terms {term_list} that directly match your query."
            elif score > 0.6:
                return f"Moderate relevance: Contains related terms {term_list} matching aspects of your query."
            else:
                return f"Some relevance: Contains terms {term_list} that have contextual connection to your query."
        else:
            # No direct term matches, but still semantically relevant
            if score > 0.8:
                return "High semantic relevance despite no direct term matches. Contains conceptually related information."
            elif score > 0.6:
                return "Moderate semantic relevance based on conceptual similarity to your query."
            else:
                return "Included based on broader contextual relevance to your query topic."
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract important terms from text, removing stopwords."""
        # Simple stopword list - could be expanded
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                    'when', 'where', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'to', 'at', 'by', 'for', 'with',
                    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                    'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under'}
        
        # Convert to lowercase, remove punctuation, and split into words
        words = ''.join([c.lower() if c.isalnum() else ' ' for c in text]).split()
        
        # Filter out stopwords and short words
        key_terms = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Return unique terms
        return list(set(key_terms))
    
    def get_context_for_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get enhanced context for a specific question.
        
        Args:
            question: Question to get context for
            top_k: Number of results to return
            
        Returns:
            Dictionary with context and metadata
        """
        # Get enhanced retrieval results
        result = self.retrieve(question, top_k=top_k)
        
        # Join content for backward compatibility
        context = "\n\n".join(result["content_list"])
        
        # Return enhanced context data
        return {
            "content": context,
            "documents": result["documents"],
            "stats": result["stats"]
        }
    
    def measure_impact(self, query: str, response: str, rag_content: List[str]) -> Dict[str, float]:
        """
        Measure the impact of RAG on a response.
        
        Args:
            query: Original query
            response: Generated response 
            rag_content: List of RAG content pieces used
            
        Returns:
            Dictionary of impact metrics
        """
        if not self.similarity_model or not rag_content:
            return {"impact_score": 0.0}
        
        try:
            # Make sure we have the util module
            from sentence_transformers import util
            
            # Encode query, response and RAG content
            query_embedding = self.similarity_model.encode(query)
            response_embedding = self.similarity_model.encode(response)
            
            # Combine RAG content and encode
            combined_rag = " ".join(rag_content)
            rag_embedding = self.similarity_model.encode(combined_rag)
            
            # Calculate similarity between response and query
            query_response_similarity = util.pytorch_cos_sim(
                query_embedding, response_embedding
            ).item()
            
            # Calculate similarity between response and RAG content
            rag_response_similarity = util.pytorch_cos_sim(
                rag_embedding, response_embedding
            ).item()
            
            # Impact is how much RAG improves over just answering the query directly
            impact_score = max(0, rag_response_similarity - query_response_similarity)
            
            return {
                "impact_score": round(impact_score, 4),
                "response_rag_similarity": round(rag_response_similarity, 4),
                "response_query_similarity": round(query_response_similarity, 4)
            }
        except Exception as e:
            print(f"Error measuring RAG impact: {e}")
            return {"impact_score": 0.0, "error": str(e)}
    
    def refresh_documents(self) -> None:
        """Refresh documents from the documents directory."""
        self.document_map = self._load_documents()
        self.questionnaire_map = self._load_questionnaires()
    
    def get_accessed_documents(self):
        """Returns list of documents accessed in the last query"""
        return self.accessed_documents
    
    def clear_accessed_documents(self):
        """Clears the list of accessed documents"""
        self.accessed_documents = []

    def get_questions_from_file(self, file_path: str) -> List[str]:
        """
        Get questions from a file.
        
        Args:
            file_path: Path to the file to get questions from
            
        Returns:
            List of questions
        """
        # If it's a JSON file, try to load questions directly
        if file_path.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'questions' in data and isinstance(data['questions'], list):
                    return data['questions']
            except Exception as e:
                raise ValueError(f"Failed to load questions from JSON file: {e}")
        
        # Otherwise, load as document and extract questions
        try:
            document = DocumentProcessor.load_document(file_path)
            if not document:
                raise ValueError(f"Failed to load document: {file_path}")
            
            questions = extract_questions_from_text(document.content)
            if not questions:
                raise ValueError(f"No questions found in document: {file_path}")
            
            return questions
        except Exception as e:
            raise ValueError(f"Error extracting questions from file {file_path}: {e}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about document retrievals."""
        # Sort top accessed documents
        sorted_docs = dict(sorted(
            self.retrieval_stats["top_accessed_documents"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])  # Top 5 documents
        
        return {
            "total_retrievals": self.retrieval_stats["total_retrievals"],
            "avg_relevance_score": round(self.retrieval_stats["avg_relevance_score"], 4),
            "top_accessed_documents": sorted_docs
        }
    
    def format_rag_citation(self, documents: List[Dict]) -> str:
        """Format RAG documents as a citation string."""
        if not documents:
            return ""
            
        citation = "\n\nSources:"
        for i, doc in enumerate(documents[:3], 1):  # Limit to top 3
            title = doc.get("title", "Unknown")
            score = doc.get("score", 0)
            citation += f"\n{i}. {title} (relevance: {score:.2f})"
            
        return citation
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        cache_stats = self.vector_store.get_cache_stats()
        doc_count = sum(len(docs) for docs in self.document_map.values())
        
        return {
            'vector_store': cache_stats,
            'total_documents': doc_count,
            'questionnaires': len(self.questionnaire_map),
            'cache_dir': str(self.vector_store.cache_dir),
            'refresh_cache': self.refresh_cache
        }