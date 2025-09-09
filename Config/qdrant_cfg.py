"""
Qdrant vector database configuration and utilities
Handles connection, collection management, and basic operations
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionStatus
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantConfig:
    """Qdrant configuration and connection manager"""
    
    def __init__(self):
        self.url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.client = None
    
    def get_client(self) -> QdrantClient:
        """Get or create Qdrant client connection"""
        if self.client is None:
            try:
                self.client = QdrantClient(url=self.url)
                logger.info(f"Connected to Qdrant at {self.url}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise
        
        return self.client
    
    def test_connection(self) -> bool:
        """Test Qdrant connection"""
        try:
            client = self.get_client()
            collections = client.get_collections()
            logger.info("Qdrant connection test successful")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection test failed: {e}")
            return False

    def ensure_collection_exists(
        self, 
        collection_name: str, 
        vector_size: int = 384,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Ensure collection exists, create if not
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (384 for all-MiniLM-L6-v2)
            distance: Distance metric for similarity search
            
        Returns:
            True if collection exists or was created successfully
        """
        try:
            client = self.get_client()
            
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name in collection_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Created collection '{collection_name}' with {vector_size} dimensions")
            return True
            
        except Exception as e:
            logger.error(f"Error managing collection '{collection_name}': {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection"""
        try:
            client = self.get_client()
            info = client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status,
                'config': info.config
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            client = self.get_client()
            client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def list_collections(self) -> list:
        """List all collections"""
        try:
            client = self.get_client()
            collections = client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

# Global instance
_qdrant_config = None

def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance"""
    global _qdrant_config
    
    if _qdrant_config is None:
        _qdrant_config = QdrantConfig()
    
    return _qdrant_config.get_client()

def ensure_collection_exists(
    collection_name: str, 
    vector_size: int = 384
) -> bool:
    """Convenience function to ensure collection exists"""
    global _qdrant_config
    
    if _qdrant_config is None:
        _qdrant_config = QdrantConfig()
    
    return _qdrant_config.ensure_collection_exists(collection_name, vector_size)

def test_qdrant_setup() -> bool:
    """Test complete Qdrant setup"""
    print("Testing Qdrant setup...")
    
    try:
        # Test connection
        config = QdrantConfig()
        if not config.test_connection():
            return False
        
        # Test collection creation
        test_collection = "test_collection_384"
        if not config.ensure_collection_exists(test_collection, 384):
            return False
        
        # Get collection info
        info = config.get_collection_info(test_collection)
        print(f"Test collection info: {info}")
        
        # List collections
        collections = config.list_collections()
        print(f"Available collections: {collections}")
        
        # Clean up test collection
        config.delete_collection(test_collection)
        
        print("Qdrant setup test passed!")
        return True
        
    except Exception as e:
        print(f"Qdrant setup test failed: {e}")
        return False

if __name__ == "__main__":
    test_qdrant_setup()