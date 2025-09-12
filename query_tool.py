#!/usr/bin/env python3
"""
Enhanced Interactive Query Tool with Full Content Retrieval
Shows complete content by combining related chunks
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Config.local_embeddings import get_local_embedding_model
from Config.qdrant_cfg import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

class EnhancedBusinessSearchInterface:
    """Enhanced search interface with full content retrieval"""
    
    def __init__(self, collection_name: str = "test_business_data"):
        """Initialize the search interface"""
        self.collection_name = collection_name
        self.embedding_model = get_local_embedding_model()
        self.qdrant_client = get_qdrant_client()
        
        # Verify collection exists
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                print(f"Error: Collection '{collection_name}' not found!")
                print(f"Available collections: {collection_names}")
                sys.exit(1)
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection(collection_name)
            self.total_points = collection_info.points_count
            
            print(f"Connected to collection '{collection_name}' with {self.total_points:,} vectors")
            
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            sys.exit(1)
    
    def search_query(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.0,
        field_filter: Optional[str] = None,
        expand_chunks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for query and optionally expand to show full content
        
        Args:
            query: User's search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            field_filter: Optional field filter
            expand_chunks: Whether to retrieve related chunks
            
        Returns:
            List of search results with expanded content
        """
        try:
            start_time = time.time()
            
            # Create query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Build filter if specified
            search_filter = None
            if field_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="field_name",
                            match=MatchValue(value=field_filter)
                        )
                    ]
                )
            
            # Perform initial search with higher limit if expanding
            search_limit = limit * 3 if expand_chunks else limit
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            if not expand_chunks:
                # Return simple results
                return self._format_simple_results(search_results, time.time() - start_time)
            
            # Group results by record and field, then expand
            grouped_results = self._group_results_by_source(search_results)
            expanded_results = self._expand_content(grouped_results, limit)
            
            search_time = time.time() - start_time
            for result in expanded_results:
                result['search_time'] = search_time
            
            return expanded_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _group_results_by_source(self, search_results) -> Dict[str, List]:
        """Group search results by record_id and field_name"""
        grouped = defaultdict(list)
        
        for result in search_results:
            record_id = result.payload.get('record_id', -1)
            field_name = result.payload.get('field_name', 'unknown')
            key = f"{record_id}_{field_name}"
            
            grouped[key].append({
                'id': result.id,
                'score': float(result.score),
                'payload': result.payload
            })
        
        # Sort chunks within each group
        for key in grouped:
            grouped[key].sort(key=lambda x: x['payload'].get('chunk_index', 0))
        
        return grouped
    
    def _expand_content(self, grouped_results: Dict, limit: int) -> List[Dict[str, Any]]:
        """Expand content by retrieving all chunks from the same field"""
        expanded_results = []
        processed_sources = set()
        
        for group_key, chunks in grouped_results.items():
            if len(expanded_results) >= limit:
                break
                
            if group_key in processed_sources:
                continue
                
            # Get the best scoring chunk from this group
            best_chunk = max(chunks, key=lambda x: x['score'])
            record_id = best_chunk['payload'].get('record_id')
            field_name = best_chunk['payload'].get('field_name')
            
            # Get all chunks from this record and field
            full_content = self._get_full_field_content(record_id, field_name)
            
            if full_content:
                expanded_result = {
                    'id': best_chunk['id'],
                    'score': best_chunk['score'],
                    'content': full_content['full_text'],
                    'header': best_chunk['payload'].get('header', ''),
                    'field_name': field_name,
                    'record_id': record_id,
                    'total_chunks': full_content['total_chunks'],
                    'matched_chunks': len(chunks),
                    'source_file': best_chunk['payload'].get('source_file', ''),
                    'content_type': 'expanded'
                }
                expanded_results.append(expanded_result)
                processed_sources.add(group_key)
        
        return expanded_results
    
    def _get_full_field_content(self, record_id: int, field_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve all chunks from a specific record and field"""
        try:
            # Search for all chunks from this record and field
            search_filter = Filter(
                must=[
                    FieldCondition(key="record_id", match=MatchValue(value=record_id)),
                    FieldCondition(key="field_name", match=MatchValue(value=field_name))
                ]
            )
            
            all_chunks = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=100,  # Should be enough for most fields
                with_payload=True,
                with_vectors=False
            )[0]
            
            if not all_chunks:
                return None
            
            # Sort chunks by chunk_index
            sorted_chunks = sorted(
                all_chunks, 
                key=lambda x: x.payload.get('chunk_index', 0)
            )
            
            # Combine all text
            full_text = " ".join(
                chunk.payload.get('text', '') for chunk in sorted_chunks
            ).strip()
            
            # Remove duplicate sentences (common in overlapping chunks)
            full_text = self._remove_duplicate_sentences(full_text)
            
            return {
                'full_text': full_text,
                'total_chunks': len(sorted_chunks)
            }
            
        except Exception as e:
            print(f"Error getting full content: {e}")
            return None
    
    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences from combined chunk text"""
        sentences = text.split('. ')
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences)
    
    def _format_simple_results(self, search_results, search_time: float) -> List[Dict[str, Any]]:
        """Format simple search results (individual chunks)"""
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
                'search_time': search_time,
                'content_type': 'chunk'
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def display_results(self, results: List[Dict[str, Any]], query: str):
        """Display search results with full content when expanded"""
        
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results for '{query}':")
        print(f"Search completed in {results[0]['search_time']:.3f} seconds")
        print("=" * 100)
        
        for i, result in enumerate(results, 1):
            content_type = result.get('content_type', 'chunk')
            
            print(f"\nResult {i}: (Score: {result['score']:.4f})")
            print(f"Header: {result['header'][:120]}...")
            print(f"Field: {result['field_name']}")
            print(f"Record ID: {result['record_id']}")
            
            if content_type == 'expanded':
                print(f"Content Type: Full Field Content")
                print(f"Matched chunks: {result.get('matched_chunks', 1)}/{result.get('total_chunks', 1)}")
                
                # Display full content with better formatting
                content = result['content']
                if len(content) > 1000:
                    print(f"\nFull Content: {content[:1000]}...")
                    print(f"\n[Content continues for {len(content) - 1000} more characters]")
                    print(f"[Total content length: {len(content)} characters]")
                else:
                    print(f"\nFull Content: {content}")
            else:
                print(f"Content Type: Individual Chunk")
                content = result['content']
                if len(content) > 400:
                    content = content[:400] + "..."
                print(f"\nContent: {content}")
                
                if result['total_chunks'] > 1:
                    print(f"Chunk: {result['chunk_index'] + 1}/{result['total_chunks']}")
            
            print("-" * 80)
    
    def interactive_search(self):
        """Enhanced interactive search with content expansion options"""
        
        print("\nEnhanced Business Data Search Interface")
        print("=" * 60)
        print(f"Collection: {self.collection_name}")
        print(f"Total vectors: {self.total_points:,}")
        print("\nSearch Modes:")
        print("  - Default: Shows full field content (expanded)")
        print("  - Add --chunks to see individual chunks only")
        print("\nCommands:")
        print("  - Enter your search query")
        print("  - 'fields' to see available fields") 
        print("  - 'help' for more options")
        print("  - 'quit' or 'exit' to quit")
        print("=" * 60)
        
        # Get available fields
        available_fields = self._get_available_fields()
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter your query: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nSearch Options:")
                    print("  query                      # Full content (default)")
                    print("  query --chunks            # Individual chunks only")
                    print("  query --limit=10          # Number of results (default: 5)")
                    print("  query --threshold=0.7     # Minimum score threshold") 
                    print("  query --field=Header_1    # Search in specific field")
                    print("  query --save              # Save results to JSON")
                    print("\nExamples:")
                    print("  'compliance procedures --limit=3'")
                    print("  'market analysis --field=Header_1 --chunks'")
                    continue
                
                elif user_input.lower() == 'fields':
                    print(f"\nAvailable fields ({len(available_fields)}):")
                    for field in available_fields:
                        print(f"  - {field}")
                    continue
                
                # Parse query and options
                query_parts = user_input.split(' --')
                query = query_parts[0].strip()
                
                # Default options
                limit = 5
                score_threshold = 0.0
                field_filter = None
                save_results = False
                expand_chunks = True  # Default to expanded content
                
                # Parse options
                for part in query_parts[1:]:
                    if part.startswith('limit='):
                        try:
                            limit = int(part.split('=')[1])
                        except:
                            print("Invalid limit value, using default (5)")
                    
                    elif part.startswith('threshold='):
                        try:
                            score_threshold = float(part.split('=')[1])
                        except:
                            print("Invalid threshold value, using default (0.0)")
                    
                    elif part.startswith('field='):
                        field_filter = part.split('=')[1]
                        if field_filter not in available_fields:
                            print(f"Warning: Field '{field_filter}' not found")
                    
                    elif part == 'save':
                        save_results = True
                    
                    elif part == 'chunks':
                        expand_chunks = False
                
                # Validate limit
                limit = max(1, min(limit, 20))
                
                # Perform search
                print(f"\nSearching for: '{query}'")
                if field_filter:
                    print(f"Field filter: {field_filter}")
                if score_threshold > 0:
                    print(f"Score threshold: {score_threshold}")
                print(f"Content mode: {'Full field content' if expand_chunks else 'Individual chunks'}")
                
                results = self.search_query(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    field_filter=field_filter,
                    expand_chunks=expand_chunks
                )
                
                # Display results
                self.display_results(results, query)
                
                # Save if requested
                if save_results and results:
                    self._save_results(results, query)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def _get_available_fields(self) -> List[str]:
        """Get list of available field names"""
        try:
            sample_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            field_names = set()
            for point in sample_points:
                field_name = point.payload.get('field_name')
                if field_name:
                    field_names.add(field_name)
            
            return sorted(list(field_names))
            
        except Exception as e:
            print(f"Error getting field names: {e}")
            return []
    
    def _save_results(self, results: List[Dict[str, Any]], query: str):
        """Save results to JSON file"""
        if not results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() or c in " -_" else "" for c in query)[:30]
        filename = f"search_results_{safe_query.replace(' ', '_')}_{timestamp}.json"
        
        export_data = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'collection': self.collection_name,
            'results': results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """Main function"""
    collection_name = "test_business_data"
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python enhanced_query_tool.py [collection_name]")
            print("\nFeatures:")
            print("- Full field content retrieval (combines chunks)")
            print("- Individual chunk viewing with --chunks")
            print("- Advanced filtering and search options")
            return
        else:
            collection_name = sys.argv[1]
    
    try:
        search_interface = EnhancedBusinessSearchInterface(collection_name)
        search_interface.interactive_search()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()