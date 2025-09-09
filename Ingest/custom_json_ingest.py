"""
Custom JSON dataset ingestion for business content
Processes JSON records with multiple text fields and creates vector embeddings
"""

import json
import os
from typing import List, Dict, Any, Optional
from Config.local_embeddings import get_local_embedding_model
from Config.qdrant_cfg import get_qdrant_client, ensure_collection_exists
from qdrant_client.models import PointStruct
import uuid
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomJSONIngestor:
    """Handles ingestion of custom JSON business dataset"""
    
    def __init__(self, embedding_model: str = None):
        """Initialize the ingestor"""
        self.embedding_model_name = embedding_model or os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = get_local_embedding_model(self.embedding_model_name)
        self.qdrant_client = get_qdrant_client()
        
        # Default text fields to process
        self.text_fields = [
            'Header_1', 'Description_1', 'Description_2', 'Description_3', 
            'Description_4', 'Description_6', 'WholeContent_1', 'WholeContent_2'
        ]
        
        logger.info(f"Initialized ingestor with model: {self.embedding_model_name}")
    
    def process_json_file(
        self,
        json_file_path: str,
        collection_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process JSON file and create embeddings
        
        Args:
            json_file_path: Path to JSON file
            collection_name: Qdrant collection name
            chunk_size: Text chunk size for embeddings
            chunk_overlap: Overlap between chunks
            max_records: Limit records for testing
            
        Returns:
            Processing summary
        """
        # Set defaults from environment if not provided
        collection_name = collection_name or os.getenv('DEFAULT_COLLECTION_NAME', 'business_dataset')
        chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', 500))
        chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', 50))
        
        logger.info(f"Processing file: {json_file_path}")
        logger.info(f"Collection: {collection_name}, Chunk size: {chunk_size}")
        
        # Ensure collection exists
        embedding_dim = self.embedding_model.get_embedding_dimension()
        if not ensure_collection_exists(collection_name, embedding_dim):
            raise Exception("Failed to create/access Qdrant collection")
        
        # Load JSON data
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_records:
            data = data[:max_records]
            logger.info(f"Limited to {max_records} records for testing")
        
        logger.info(f"Loaded {len(data)} records")
        
        # Process records
        all_texts, all_metadata = self._extract_texts_from_records(
            data, json_file_path, chunk_size, chunk_overlap
        )
        
        # Create embeddings
        embeddings = self._create_embeddings(all_texts)
        
        # Upload to Qdrant
        points_created = self._upload_to_qdrant(embeddings, all_metadata, collection_name)
        
        # Generate summary
        summary = {
            'source_file': json_file_path,
            'total_records': len(data),
            'total_chunks': len(all_texts),
            'points_created': points_created,
            'collection_name': collection_name,
            'embedding_model': self.embedding_model_name,
            'embedding_dimensions': embedding_dim
        }
        
        logger.info("Processing complete!")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return summary
    
    def _extract_texts_from_records(
        self, 
        data: List[Dict], 
        source_file: str,
        chunk_size: int, 
        chunk_overlap: int
    ) -> tuple:
        """Extract and chunk text from JSON records"""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        all_texts = []
        all_metadata = []
        
        for record_idx, record in enumerate(tqdm(data, desc="Extracting texts")):
            
            for field_name in self.text_fields:
                if field_name not in record or not record[field_name]:
                    continue
                
                content = str(record[field_name]).strip()
                
                # Skip very short content
                if len(content) < 20:
                    continue
                
                # Split long content into chunks
                if len(content) > chunk_size:
                    chunks = text_splitter.split_text(content)
                else:
                    chunks = [content]
                
                # Create metadata for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_metadata.append({
                        'record_id': record_idx,
                        'field_name': field_name,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'text': chunk,
                        'header': record.get('Header_1', 'Unknown')[:100],
                        'content_type': 'business_content',
                        'source_file': os.path.basename(source_file),
                        'chunk_length': len(chunk)
                    })
        
        logger.info(f"Extracted {len(all_texts)} text chunks")
        return all_texts, all_metadata
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for texts"""
        batch_size = int(os.getenv('BATCH_SIZE', 32))
        
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.embed_documents(
            texts,
            batch_size=batch_size,
            show_progress=True
        )
        
        return embeddings
    
    def _upload_to_qdrant(
        self, 
        embeddings: List[List[float]], 
        metadata_list: List[Dict], 
        collection_name: str
    ) -> int:
        """Upload points to Qdrant"""
        
        # Create points
        points = []
        for embedding, metadata in zip(embeddings, metadata_list):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=metadata
            )
            points.append(point)
        
        # Upload in batches
        upload_batch_size = 100
        total_uploaded = 0
        
        logger.info(f"Uploading {len(points)} points to Qdrant...")
        for i in tqdm(range(0, len(points), upload_batch_size), desc="Uploading"):
            batch = points[i:i + upload_batch_size]
            
            try:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                total_uploaded += len(batch)
            except Exception as e:
                logger.error(f"Failed to upload batch {i}: {e}")
        
        return total_uploaded

def test_small_subset(json_file: str = "Data/new_cleaned_data.json", max_records: int = None):
    """Test ingestion with small subset"""
    
    print("Testing JSON ingestion with small subset...")
    
    try:
        ingestor = CustomJSONIngestor()
        summary = ingestor.process_json_file(
            json_file_path=json_file,
            collection_name="test_business_data",
            max_records=max_records
        )
        
        print("Test completed successfully!")
        print("Summary:", summary)
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def process_full_dataset(json_file: str = "Data/new_cleaned_data.json"):
    """Process complete dataset"""
    
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        print("Make sure to copy your JSON file to the Data/ directory")
        return False
    
    print("This will process the complete dataset...")
    response = input("Continue? (y/N): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        return False
    
    try:
        ingestor = CustomJSONIngestor()
        summary = ingestor.process_json_file(
            json_file_path=json_file,
            collection_name="business_dataset_full"
        )
        
        print("Full dataset processing complete!")
        return summary
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return False

if __name__ == "__main__":
    # Run test by default
    test_small_subset()