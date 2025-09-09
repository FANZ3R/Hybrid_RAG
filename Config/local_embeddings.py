"""
Local embedding model using sentence-transformers
Provides cost-effective alternative to API-based embeddings
"""

import os
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingModel:
    """Local embedding model using sentence-transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize local embedding model
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        if model_name is None:
            model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Create embeddings for text(s)
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Single embedding or list of embeddings
        """
        try:
            if isinstance(text, str):
                embedding = self.model.encode(text)
                return embedding.tolist()
            else:
                embeddings = self.model.encode(text)
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    def embed_documents(
        self, 
        documents: List[str], 
        batch_size: int = None, 
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple documents efficiently
        
        Args:
            documents: List of text documents
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings
        """
        if batch_size is None:
            batch_size = int(os.getenv('BATCH_SIZE', 32))
            
        try:
            embeddings = self.model.encode(
                documents, 
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = self.model.similarity(embeddings[0], embeddings[1])
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            raise

# Global instance
_embedding_model = None

def get_local_embedding_model(model_name: str = None) -> LocalEmbeddingModel:
    """
    Get or create singleton embedding model instance
    
    Args:
        model_name: Model name (uses env variable if None)
        
    Returns:
        LocalEmbeddingModel instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = LocalEmbeddingModel(model_name)
    
    return _embedding_model

def test_local_embeddings() -> bool:
    """Test local embedding functionality"""
    print("Testing local embeddings...")
    
    try:
        # Initialize model
        embedding_model = get_local_embedding_model()
        
        # Test single text
        text = "This is a test sentence for embedding."
        embedding = embedding_model.embed_text(text)
        print(f"Single text embedding: {len(embedding)} dimensions")
        
        # Test multiple texts
        texts = [
            "Supplier management systems help businesses.",
            "Onion prices are affected by weather conditions.",
            "Local embedding models provide cost savings."
        ]
        embeddings = embedding_model.embed_documents(texts, show_progress=False)
        print(f"Batch embeddings: {len(embeddings)} texts, {len(embeddings[0])} dimensions each")
        
        # Test similarity
        similarity = embedding_model.similarity(texts[0], texts[1])
        print(f"Similarity between different topics: {similarity:.3f}")
        
        similarity = embedding_model.similarity(texts[0], "Business management solutions")
        print(f"Similarity between related topics: {similarity:.3f}")
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_local_embeddings()