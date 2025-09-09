"""
Hybrid Search Implementation for Business Data RAG
Combines semantic vector search with filtering and ranking
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Config.local_embeddings import get_local_embedding_model
from Config.qdrant_cfg import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """
    Hybrid search engine combining vector similarity with metadata filtering
    """
    
    def __init__(self, collection_name: str = "business_dataset"):
        """
        Initialize the hybrid search engine
        
        Args:
            collection_name: Name of the Qdrant collection to search
        """
        self.collection_name = collection_name
        self.embedding_model = get_local_embedding_model()
        self.qdrant_client = get_qdrant_client()
        
        logger.info(f"Initialized HybridSearchEngine for collection: {collection_name}")
    
    def semantic_search(
        self, 
        query: str, 
        limit: int = 10,
        score_threshold: float = 0.0,
        field_filter: Optional[str] = None,
        record_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic vector search
        
        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            field_filter: Filter by specific field name
            record_filter: Filter by specific record ID
            
        Returns:
            List of search results with metadata
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Build filter conditions
            filter_conditions = []
            
            if field_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="field_name",
                        match=MatchValue(value=field_filter)
                    )
                )
            
            if record_filter is not None:
                filter_conditions.append(
                    FieldCondition(
                        key="record_id",
                        match=MatchValue(value=record_filter)
                    )
                )
            
            # Create filter object
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    'id': result.id,
                    'score': float(result.score),
                    'content': result.payload.get('text', ''),
                    'header': result.payload.get('header', ''),
                    'field_name': result.payload.get('field_name', ''),
                    'record_id': result.payload.get('record_id', -1),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'total_chunks': result.payload.get('total_chunks', 1),
                    'source_file': result.payload.get('source_file', ''),
                    'chunk_length': result.payload.get('chunk_length', 0)
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def search_by_field(
        self, 
        query: str, 
        field_name: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search within specific field only
        
        Args:
            query: Search query
            field_name: Field to search in (e.g., 'Header_1', 'Description_1')
            limit: Maximum results
            
        Returns:
            List of search results
        """
        return self.semantic_search(
            query=query,
            limit=limit,
            field_filter=field_name
        )
    
    def search_by_record(
        self, 
        query: str, 
        record_id: int, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search within specific record only
        
        Args:
            query: Search query
            record_id: Record ID to search in
            limit: Maximum results
            
        Returns:
            List of search results
        """
        return self.semantic_search(
            query=query,
            limit=limit,
            record_filter=record_id
        )
    
    def multi_query_search(
        self, 
        queries: List[str], 
        limit_per_query: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search multiple queries and return aggregated results
        
        Args:
            queries: List of search queries
            limit_per_query: Limit per individual query
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            results[query] = self.semantic_search(
                query=query,
                limit=limit_per_query
            )
        
        return results
    
    def get_similar_content(
        self, 
        reference_id: str, 
        limit: int = 5,
        exclude_same_record: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find content similar to a reference point
        
        Args:
            reference_id: ID of reference point
            limit: Maximum results
            exclude_same_record: Whether to exclude results from same record
            
        Returns:
            List of similar content
        """
        try:
            # Get the reference point
            reference_points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[reference_id],
                with_vectors=True,
                with_payload=True
            )
            
            if not reference_points:
                logger.warning(f"Reference point {reference_id} not found")
                return []
            
            reference_point = reference_points[0]
            reference_vector = reference_point.vector
            
            # Build filter to exclude same record if requested
            search_filter = None
            if exclude_same_record and reference_point.payload:
                record_id = reference_point.payload.get('record_id')
                if record_id is not None:
                    search_filter = Filter(
                        must_not=[
                            FieldCondition(
                                key="record_id",
                                match=MatchValue(value=record_id)
                            )
                        ]
                    )
            
            # Search for similar content
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=reference_vector,
                limit=limit + 1,  # +1 to account for the reference itself
                query_filter=search_filter,
                with_payload=True
            )
            
            # Format and filter out the reference point itself
            similar_content = []
            for result in search_results:
                if result.id != reference_id:
                    formatted_result = {
                        'id': result.id,
                        'score': float(result.score),
                        'content': result.payload.get('text', ''),
                        'header': result.payload.get('header', ''),
                        'field_name': result.payload.get('field_name', ''),
                        'record_id': result.payload.get('record_id', -1)
                    }
                    similar_content.append(formatted_result)
            
            return similar_content[:limit]
            
        except Exception as e:
            logger.error(f"Similar content search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Get sample of payloads to analyze field distribution
            sample_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Analyze field distribution
            field_counts = {}
            record_counts = {}
            
            for point in sample_points:
                field_name = point.payload.get('field_name', 'unknown')
                record_id = point.payload.get('record_id', -1)
                
                field_counts[field_name] = field_counts.get(field_name, 0) + 1
                record_counts[record_id] = record_counts.get(record_id, 0) + 1
            
            stats = {
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'collection_status': collection_info.status.value,
                'sample_field_distribution': field_counts,
                'sample_record_distribution': record_counts,
                'indexed_vectors': collection_info.indexed_vectors_count
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

def test_hybrid_search(collection_name: str = "test_business_data"):
    """
    Test the hybrid search functionality
    
    Args:
        collection_name: Name of collection to test
    """
    print("🚀 Testing Hybrid Search Engine")
    print("=" * 80)
    
    try:
        # Initialize search engine
        search_engine = HybridSearchEngine(collection_name)
        
        # Test 1: Basic semantic search
        print("\n📋 Test 1: Basic Semantic Search")
        print("-" * 40)
        
        test_queries = [
            "silica market prices",
            "isobutanol production process", 
            "supply chain container shortages",
            "construction sector trends"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = search_engine.semantic_search(query, limit=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.4f}")
                print(f"     Field: {result['field_name']}")
                print(f"     Content: {result['content'][:100]}...")
                print(f"     Header: {result['header'][:60]}...")
        
        # Test 2: Field-specific search
        print(f"\n📋 Test 2: Field-Specific Search")
        print("-" * 40)
        
        field_results = search_engine.search_by_field(
            query="market analysis", 
            field_name="Header_1", 
            limit=3
        )
        
        print(f"Found {len(field_results)} results in 'Header_1' field:")
        for result in field_results:
            print(f"  - {result['header'][:80]}...")
        
        # Test 3: Collection statistics
        print(f"\n📋 Test 3: Collection Statistics")
        print("-" * 40)
        
        stats = search_engine.get_collection_stats()
        print(f"Total Points: {stats.get('total_points', 'N/A')}")
        print(f"Vector Size: {stats.get('vector_size', 'N/A')}")
        print(f"Distance Metric: {stats.get('distance_metric', 'N/A')}")
        print(f"Collection Status: {stats.get('collection_status', 'N/A')}")
        
        if stats.get('sample_field_distribution'):
            print("\nField Distribution (sample):")
            for field, count in stats['sample_field_distribution'].items():
                print(f"  - {field}: {count}")
        
        print("\n✅ Hybrid search testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search testing failed: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    test_hybrid_search()