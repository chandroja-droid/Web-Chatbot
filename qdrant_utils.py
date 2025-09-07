from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QdrantManager:
    """Manages Qdrant Cloud vector database operations with consistent dimensions"""
    
    def __init__(self, vector_size: int = 384):  # DEFAULT TO 384 FOR LOCAL EMBEDDINGS
        self.collection_name = "document_chunks"
        self.vector_size = vector_size  # Use consistent size
        
        # Connect to Qdrant Cloud
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30
        )
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,  # Use instance size
                        distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created collection: {self.collection_name} with vector size {self.vector_size}")
            else:
                # Check if existing collection has the right vector size
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                
                if existing_size != self.vector_size:
                    logger.warning(f"Collection size mismatch! Expected {self.vector_size}, got {existing_size}")
                    # Auto-fix: Recreate collection with correct size
                    logger.info("Auto-fixing: Recreating collection with correct size...")
                    self.client.delete_collection(collection_name=self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        ),
                    )
                    logger.info(f"Recreated collection with correct size: {self.vector_size}")
                else:
                    logger.info(f"Collection {self.collection_name} already exists with correct size {existing_size}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise ConnectionError(f"Failed to connect to Qdrant Cloud: {e}")
    
    async def store_chunks(self, chunks: List[Dict[str, Any]], source: str, file_type: str) -> int:
        """Store document chunks in Qdrant Cloud with dimension validation"""
        if not chunks:
            return 0
        
        points = []
        valid_chunks = 0
        
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            embedding = chunk.get("embedding", [])
            
            # Validate embedding dimension
            if len(embedding) != self.vector_size:
                logger.error(f"❌ CRITICAL: Embedding dimension mismatch! Expected {self.vector_size}, got {len(embedding)}")
                logger.error(f"   Source: {source}, File type: {file_type}")
                logger.error(f"   This should never happen with consistent embedding service")
                continue  # Skip invalid chunks
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        "source": source,
                        "file_type": file_type,
                        "chunk_id": point_id,
                        "start_index": chunk.get("start_index", 0),
                        "end_index": chunk.get("end_index", 0),
                        "timestamp": datetime.now().isoformat(),
                        "chunk_size": len(chunk["content"]),
                        "embedding_size": len(embedding)
                    }
                )
            )
            valid_chunks += 1
        
        if not points:
            logger.warning("No valid chunks to store")
            return 0
        
        try:
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"✅ Stored {len(points)} chunks from {source} (file type: {file_type})")
            return len(points)
            
        except Exception as e:
            logger.error(f"Error storing chunks in Qdrant: {e}")
            # Provide more detailed error information
            if "dimension error" in str(e):
                logger.error("💥 Dimension error detected! Please check your embedding service configuration.")
            raise ConnectionError(f"Failed to store data in Qdrant: {e}")
    
    async def search_similar(self, query_embedding: List[float], limit: int = 5, 
                           source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            # Validate query embedding dimension
            if len(query_embedding) != self.vector_size:
                logger.error(f"Query embedding dimension mismatch: got {len(query_embedding)}, expected {self.vector_size}")
                # Try to handle gracefully by padding/truncating
                if len(query_embedding) < self.vector_size:
                    query_embedding = query_embedding + [0.0] * (self.vector_size - len(query_embedding))
                else:
                    query_embedding = query_embedding[:self.vector_size]
                logger.warning(f"Adjusted query embedding to size {len(query_embedding)}")
            
            # Build filter if source is specified
            query_filter = None
            if source_filter:
                query_filter = Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
                )
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                score_threshold=0.3
            )
            
            return [
                {
                    "score": result.score,
                    "content": result.payload["content"],
                    "source": result.payload["source"],
                    "file_type": result.payload.get("file_type", "unknown"),
                    "chunk_id": result.payload["chunk_id"],
                    "timestamp": result.payload.get("timestamp")
                }
                for result in search_results
            ]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise ConnectionError(f"Search failed: {e}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "expected_size": self.vector_size,
                "status": "connected",
                "size_match": collection_info.config.params.vectors.size == self.vector_size
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"status": "disconnected", "error": str(e)}
    
    async def check_connection(self) -> bool:
        """Check if Qdrant Cloud connection is working"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {e}")
            return False

    async def recreate_collection(self, new_vector_size: int = None):
        """Recreate the collection with a new vector size"""
        try:
            if new_vector_size is None:
                new_vector_size = self.vector_size
                
            # Delete existing collection
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
            # Update vector size
            self.vector_size = new_vector_size
            
            # Create new collection
            self._ensure_collection()
            logger.info(f"Recreated collection with vector size {new_vector_size}")
            
            return True
        except Exception as e:
            logger.error(f"Error recreating collection: {e}")
            return False
    
    async def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info("Collection cleared and recreated")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

# Global Qdrant manager instance - will be initialized with correct vector size
qdrant_manager = None

def initialize_qdrant_manager(vector_size: int = 384):  # DEFAULT TO 384
    """Initialize the Qdrant manager with the correct vector size"""
    global qdrant_manager
    qdrant_manager = QdrantManager(vector_size)
    return qdrant_manager
