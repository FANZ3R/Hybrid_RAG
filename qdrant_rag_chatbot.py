import streamlit as st
from openai import OpenAI
import re
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Config.local_embeddings import get_local_embedding_model
from Config.qdrant_cfg import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from collections import defaultdict

# ========== CONFIG ==========
OPENROUTER_API_KEY = "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"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== VECTOR SEARCH ENGINE ==========
class QdrantRAGEngine:
    """RAG engine using Qdrant vector database"""
    
    def __init__(self, collection_name: str = "test_business_data"):
        self.collection_name = collection_name
        self.embedding_model = get_local_embedding_model()
        self.qdrant_client = get_qdrant_client()
        
        # Verify collection exists
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                st.error(f"Collection '{collection_name}' not found!")
                st.error(f"Available collections: {collection_names}")
                return
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection(collection_name)
            self.total_points = collection_info.points_count
            
            logger.info(f"Connected to collection '{collection_name}' with {self.total_points:,} vectors")
            
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            st.error(f"Error connecting to Qdrant: {e}")
            return
    
    def search_for_context(
        self, 
        query: str, 
        limit: int = 10,
        score_threshold: float = 0.3,
        expand_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using multiple strategies
        
        Args:
            query: User's question
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            expand_context: Whether to expand chunks to full field content
            
        Returns:
            List of context documents
        """
        all_contexts = []
        
        try:
            # Strategy 1: Direct semantic search
            contexts = self._semantic_search(query, limit, score_threshold, expand_context)
            all_contexts.extend(contexts)
            
            # Strategy 2: Multi-query approach (break down complex questions)
            if len(all_contexts) < 5:
                sub_queries = self._generate_sub_queries(query)
                for sub_query in sub_queries[:2]:  # Limit to 2 sub-queries
                    sub_contexts = self._semantic_search(sub_query, 5, score_threshold, expand_context)
                    all_contexts.extend(sub_contexts)
            
            # Strategy 3: Field-specific search for certain query types
            if len(all_contexts) < 5:
                field_contexts = self._field_specific_search(query, limit//2, score_threshold)
                all_contexts.extend(field_contexts)
            
            # Remove duplicates and sort by relevance
            unique_contexts = self._deduplicate_contexts(all_contexts)
            
            return unique_contexts[:limit]
            
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            return []
    
    def _semantic_search(
        self, 
        query: str, 
        limit: int, 
        score_threshold: float,
        expand_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform semantic vector search"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit * 2,  # Get more results for processing
                score_threshold=score_threshold,
                with_payload=True
            )
            
            if not expand_context:
                # Return individual chunks
                return [self._format_chunk_context(result) for result in search_results[:limit]]
            
            # Group and expand content
            grouped_results = self._group_results_by_source(search_results)
            expanded_contexts = self._expand_context_content(grouped_results, limit)
            
            return expanded_contexts
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
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
    
    def _expand_context_content(self, grouped_results: Dict, limit: int) -> List[Dict[str, Any]]:
        """Expand context by combining chunks from same field"""
        expanded_contexts = []
        processed_sources = set()
        
        # Sort groups by best score
        sorted_groups = sorted(
            grouped_results.items(),
            key=lambda x: max(chunk['score'] for chunk in x[1]),
            reverse=True
        )
        
        for group_key, chunks in sorted_groups:
            if len(expanded_contexts) >= limit:
                break
                
            if group_key in processed_sources:
                continue
            
            # Get the best scoring chunk from this group
            best_chunk = max(chunks, key=lambda x: x['score'])
            record_id = best_chunk['payload'].get('record_id')
            field_name = best_chunk['payload'].get('field_name')
            
            # Get full field content
            full_content = self._get_full_field_content(record_id, field_name)
            
            if full_content:
                context = {
                    'content': full_content['full_text'],
                    'header': best_chunk['payload'].get('header', ''),
                    'field_name': field_name,
                    'record_id': record_id,
                    'score': best_chunk['score'],
                    'matched_chunks': len(chunks),
                    'total_chunks': full_content['total_chunks'],
                    'source_file': best_chunk['payload'].get('source_file', ''),
                    'context_type': 'expanded'
                }
                expanded_contexts.append(context)
                processed_sources.add(group_key)
        
        return expanded_contexts
    
    def _get_full_field_content(self, record_id: int, field_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve all chunks from a specific record and field"""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(key="record_id", match=MatchValue(value=record_id)),
                    FieldCondition(key="field_name", match=MatchValue(value=field_name))
                ]
            )
            
            all_chunks = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if not all_chunks:
                return None
            
            # Sort chunks by index
            sorted_chunks = sorted(
                all_chunks, 
                key=lambda x: x.payload.get('chunk_index', 0)
            )
            
            # Combine text
            full_text = " ".join(
                chunk.payload.get('text', '') for chunk in sorted_chunks
            ).strip()
            
            # Clean up duplicate sentences
            full_text = self._remove_duplicate_sentences(full_text)
            
            return {
                'full_text': full_text,
                'total_chunks': len(sorted_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error getting full content: {e}")
            return None
    
    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences from combined text"""
        sentences = text.split('. ')
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences)
    
    def _format_chunk_context(self, result) -> Dict[str, Any]:
        """Format individual chunk as context"""
        return {
            'content': result.payload.get('text', ''),
            'header': result.payload.get('header', ''),
            'field_name': result.payload.get('field_name', ''),
            'record_id': result.payload.get('record_id', -1),
            'score': float(result.score),
            'chunk_index': result.payload.get('chunk_index', 0),
            'total_chunks': result.payload.get('total_chunks', 1),
            'source_file': result.payload.get('source_file', ''),
            'context_type': 'chunk'
        }
    
    def _field_specific_search(self, query: str, limit: int, score_threshold: float) -> List[Dict[str, Any]]:
        """Search in specific fields based on query type"""
        contexts = []
        
        # Determine likely relevant fields based on query keywords
        query_lower = query.lower()
        target_fields = []
        
        if any(word in query_lower for word in ['title', 'header', 'name', 'what is']):
            target_fields.append('Header_1')
        
        if any(word in query_lower for word in ['description', 'explain', 'details', 'about']):
            target_fields.extend(['Description_1', 'Description_2', 'Description_3', 'Description_4'])
        
        if any(word in query_lower for word in ['content', 'full', 'complete', 'comprehensive']):
            target_fields.extend(['WholeContent_1', 'WholeContent_2'])
        
        # Search in each target field
        for field in target_fields[:3]:  # Limit to 3 fields
            try:
                field_contexts = self._semantic_search_with_field_filter(
                    query, field, limit//len(target_fields) + 1, score_threshold
                )
                contexts.extend(field_contexts)
            except Exception as e:
                logger.warning(f"Field search failed for {field}: {e}")
                continue
        
        return contexts
    
    def _semantic_search_with_field_filter(
        self, query: str, field_name: str, limit: int, score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search with field filter"""
        try:
            query_embedding = self.embedding_model.embed_text(query)
            
            search_filter = Filter(
                must=[FieldCondition(key="field_name", match=MatchValue(value=field_name))]
            )
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            return [self._format_chunk_context(result) for result in search_results]
            
        except Exception as e:
            logger.error(f"Field-filtered search failed: {e}")
            return []
    
    def _generate_sub_queries(self, query: str) -> List[str]:
        """Generate related sub-queries for better coverage"""
        # Simple rule-based sub-query generation
        sub_queries = []
        
        # Extract key terms
        key_terms = self._extract_key_terms(query)
        
        if len(key_terms) > 1:
            # Create queries with individual terms
            for term in key_terms[:2]:
                sub_queries.append(term)
        
        # Add domain-specific variations
        query_lower = query.lower()
        if 'compliance' in query_lower:
            sub_queries.extend(['regulatory requirements', 'audit procedures'])
        elif 'risk' in query_lower:
            sub_queries.extend(['risk assessment', 'risk management'])
        elif 'supplier' in query_lower:
            sub_queries.extend(['vendor management', 'procurement'])
        elif 'process' in query_lower:
            sub_queries.extend(['workflow', 'procedures'])
        
        return sub_queries[:3]  # Limit to 3 sub-queries
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        stopwords = {
            'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may'
        }
        
        words = re.findall(r'\w+', query.lower())
        key_terms = [w for w in words if w not in stopwords and len(w) > 3]
        
        return key_terms
    
    def _deduplicate_contexts(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate contexts and sort by relevance"""
        seen_content = set()
        unique_contexts = []
        
        # Sort by score first
        contexts.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        for context in contexts:
            # Create a simple hash of the content for deduplication
            content_hash = hash(context.get('content', '')[:200])  # First 200 chars
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_contexts.append(context)
        
        return unique_contexts
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Sample points for field analysis
            sample_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            field_distribution = defaultdict(int)
            record_distribution = defaultdict(int)
            
            for point in sample_points:
                field_name = point.payload.get('field_name', 'unknown')
                record_id = point.payload.get('record_id', -1)
                
                field_distribution[field_name] += 1
                record_distribution[record_id] += 1
            
            return {
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'field_distribution': dict(field_distribution),
                'sample_records': len(record_distribution),
                'avg_chunks_per_record': sum(record_distribution.values()) / len(record_distribution) if record_distribution else 0
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

# ========== OPENROUTER CLIENT ==========
@st.cache_resource
def init_openai_client():
    """Initialize OpenAI client with caching"""
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

@st.cache_resource
def init_rag_engine(collection_name: str = "test_business_data"):
    """Initialize RAG engine with caching"""
    return QdrantRAGEngine(collection_name)

client = init_openai_client()

def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Generate answer using Llama3 model with enhanced prompting"""
    if not contexts:
        context_text = "No relevant information found in the knowledge base."
        context_summary = "No context available"
    else:
        # Format contexts for the prompt
        context_parts = []
        for i, ctx in enumerate(contexts[:8], 1):  # Limit to top 8 contexts
            header = ctx.get('header', 'N/A')[:100]
            content = ctx.get('content', '')[:500]  # Limit content length
            field = ctx.get('field_name', 'unknown')
            score = ctx.get('score', 0)
            
            context_parts.append(
                f"[Context {i}] (Relevance: {score:.2f})\n"
                f"Source: {field} - {header}\n"
                f"Content: {content}\n"
            )
        
        context_text = "\n".join(context_parts)
        context_summary = f"{len(contexts)} relevant documents found"
    
    prompt = f"""
You are an AI assistant with access to a comprehensive business knowledge base containing information about business processes, compliance procedures, risk management, supplier relationships, and organizational operations.

User Question: {question}

Relevant Information from Knowledge Base:
{context_text}

Instructions:
1. Provide a clear, comprehensive answer based on the available information
2. If the information directly answers the question, synthesize it into a coherent response
3. If the information is partially relevant, use it to provide the best possible answer and explain what aspects are covered
4. When referencing information, mention the source context (e.g., "According to the compliance documentation...")
5. If multiple sources provide related information, combine them logically
6. If no relevant information is found, suggest how the question might be rephrased or what related topics might be available
7. Keep your response informative but concise (aim for 2-4 paragraphs)

Context Summary: {context_summary}
"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"I encountered an error while generating the response: {e}"

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="Vector RAG Chatbot", 
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Business Knowledge RAG Chatbot")
st.caption("Ask questions about your business data using vector semantic search")

# Initialize RAG engine
collection_name = st.sidebar.selectbox(
    "Select Collection",
    ["test_business_data", "business_dataset_full"],
    index=0
)

rag_engine = init_rag_engine(collection_name)

# Sidebar with statistics
with st.sidebar:
    st.header("📊 Collection Statistics")
    with st.spinner("Loading statistics..."):
        stats = rag_engine.get_collection_stats() if rag_engine else {}
    
    if stats:
        st.metric("Total Vectors", f"{stats.get('total_points', 0):,}")
        st.metric("Vector Dimensions", stats.get('vector_size', 0))
        st.metric("Distance Metric", stats.get('distance_metric', 'N/A'))
        
        if stats.get('field_distribution'):
            st.subheader("Field Distribution")
            for field, count in sorted(stats['field_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                st.text(f"• {field}: {count}")
        
        avg_chunks = stats.get('avg_chunks_per_record', 0)
        if avg_chunks > 0:
            st.metric("Avg Chunks/Record", f"{avg_chunks:.1f}")
    
    st.subheader("💡 Try asking about:")
    st.text("• What are compliance procedures?")
    st.text("• How does risk management work?")
    st.text("• Supplier evaluation processes")
    st.text("• Business automation tools")
    st.text("• Market analysis reports")
    
    # Search settings
    st.subheader("⚙️ Search Settings")
    search_limit = st.slider("Max Results", 3, 15, 8)
    score_threshold = st.slider("Score Threshold", 0.0, 0.8, 0.3, 0.1)
    expand_content = st.checkbox("Expand to Full Content", True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask me about your business data...")

if user_input and rag_engine:
    # Save user input
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Searching knowledge base..."):
        start_time = time.time()
        
        # Search for relevant contexts
        contexts = rag_engine.search_for_context(
            query=user_input,
            limit=search_limit,
            score_threshold=score_threshold,
            expand_context=expand_content
        )
        
        search_time = time.time() - start_time
        
        st.info(f"🔍 Found {len(contexts)} relevant contexts in {search_time:.2f}s")
    
    with st.spinner("Generating response..."):
        # Generate answer
        answer = generate_answer(user_input, contexts)
    
    # Save bot answer
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Show found contexts in expander
    if contexts:
        with st.expander(f"📋 Found {len(contexts)} relevant contexts (click to expand)"):
            for i, ctx in enumerate(contexts, 1):
                score = ctx.get('score', 0)
                header = ctx.get('header', 'N/A')
                field = ctx.get('field_name', 'unknown')
                content = ctx.get('content', '')[:300]
                
                st.markdown(f"**Context {i}** (Score: {score:.3f})")
                st.markdown(f"*Source: {field} - {header[:80]}...*")
                st.text(content + ("..." if len(ctx.get('content', '')) > 300 else ""))
                st.divider()

elif user_input and not rag_engine:
    st.error("Cannot process questions - Qdrant connection failed. Please check your configuration.")

# Display the chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Connection status
if not rag_engine:
    st.error("⚠️ Qdrant connection failed. Please check your configuration and ensure Qdrant is running.")
    st.text("Troubleshooting:")
    st.text("1. Make sure Qdrant is running (docker-compose up -d)")
    st.text("2. Verify the collection exists and has data")
    st.text("3. Check if port 6333 is accessible")
else:
    st.success("✅ Connected to Qdrant vector database")