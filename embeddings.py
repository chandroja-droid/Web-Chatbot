# embeddings.py - Fixed version
import openai
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
import os
import asyncio
from dotenv import load_dotenv
from fastapi import HTTPException
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EmbeddingService:
    """Handles text embedding generation with fallback options"""
    
class EmbeddingService:
    """Handles text embedding generation - LOCAL ONLY MODE"""
    
    def __init__(self):
        # Force local embeddings only
        self.local_model = None
        self.embedding_size = 384  # Default size for local embeddings
        
        # Initialize local model
        try:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_size = self.local_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Local embedding model loaded with size: {self.embedding_size}")
        except Exception as e:
            logger.error(f"❌ Failed to load local embedding model: {e}")
            raise HTTPException(status_code=500, detail="Local embedding model failed to load")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using local model"""
        if not text.strip():
            return [0.0] * self.embedding_size
        
        # Always use local model
        return await self._get_local_embedding(text)
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using local model"""
        if not texts:
            return []
        
        # Always use local model
        return await self._get_local_embeddings_batch(texts)
    
    async def _get_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model"""
        try:
            embedding = self.local_model.encode(text)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise HTTPException(status_code=500, detail="Local embedding generation failed")
    
    async def _get_local_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using local model"""
        try:
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text.strip()]
            if not non_empty_texts:
                return [[] for _ in texts]
            
            # Encode the non-empty texts
            embeddings = self.local_model.encode(non_empty_texts)
            
            # Normalize each embedding
            normalized_embeddings = []
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized_emb = emb / norm
                else:
                    normalized_emb = emb
                normalized_embeddings.append(normalized_emb.tolist())
            
            # Map back to original text positions
            result = []
            text_index = 0
            
            for text in texts:
                if text.strip():
                    result.append(normalized_embeddings[text_index])
                    text_index += 1
                else:
                    result.append([0.0] * self.embedding_size)
            
            return result
            
        except Exception as e:
            logger.error(f"Local batch embedding failed: {e}")
            raise HTTPException(status_code=500, detail="Local batch embedding failed")


    async def get_embeddings_batch(self, texts: List[str], use_local: bool = True, max_retries: int = 3) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        if not texts:
            return []
        
        # Use local model for batch processing if requested
        if use_local and self.local_model:
            return await self._get_local_embeddings_batch(texts)
        
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            return [[] for _ in texts]
        
        # Simple retry logic without tenacity
        for attempt in range(max_retries):
            try:
                # OLD SYNTAX: Use openai.Embedding.create
                response = openai.Embedding.create(
                    input=non_empty_texts,
                    model=self.model
                )
                
                # OLD SYNTAX: Different response structure
                embeddings = [item['embedding'] for item in response['data']]
                result = []
                text_index = 0
                
                for text in texts:
                    if text.strip():
                        result.append(embeddings[text_index])
                        text_index += 1
                    else:
                        result.append([0.0] * self.embedding_size)
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.warning(f"OpenAI batch embedding failed after {max_retries} attempts, using local model: {e}")
                    if self.local_model:
                        return await self._get_local_embeddings_batch(texts)
                    raise HTTPException(status_code=500, detail="Batch embedding failed")
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"OpenAI batch attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def _get_local_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using local model - FIXED VERSION"""
        try:
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text.strip()]
            if not non_empty_texts:
                return [[] for _ in texts]
            
            # Encode the non-empty texts
            embeddings = self.local_model.encode(non_empty_texts)
            
            # Normalize each embedding and convert to list
            normalized_embeddings = []
            for emb in embeddings:
                normalized_emb = emb / np.linalg.norm(emb)
                normalized_embeddings.append(normalized_emb.tolist())
            
            # Map back to original text positions
            result = []
            text_index = 0
            
            for text in texts:
                if text.strip():
                    result.append(normalized_embeddings[text_index])
                    text_index += 1
                else:
                    result.append([0.0] * self.embedding_size)
            
            return result
            
        except Exception as e:
            logger.error(f"Local batch embedding failed: {e}")
            raise HTTPException(status_code=500, detail="Local batch embedding failed")

# Global embedding service instance
embedding_service = EmbeddingService()
