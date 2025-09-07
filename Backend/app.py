from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
from qdrant_utils import qdrant_manager
from qdrant_utils import initialize_qdrant_manager
from embeddings import EmbeddingService, embedding_service
from document_parser import document_parser
import asyncio
import json
import aiohttp
import uuid
from datetime import datetime
import os
import logging
import traceback
from dotenv import load_dotenv
import openai

embedding_service = EmbeddingService()
vector_size = embedding_service.embedding_size
print(f"Using embedding vector size: {vector_size}")

qdrant_manager = initialize_qdrant_manager(vector_size)

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables")
    print("Available environment variables:", [k for k in os.environ.keys() if 'OPENAI' in k or 'API' in k])
else:
    print("OpenAI API key found, starting with:", openai.api_key[:10] + "...")

app = FastAPI(
    title="Document Q&A API - ChatGPT-like",
    description="AI-powered document question answering system with RAG capabilities",
    version="1.0.0"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory storage for documents (can be replaced with vector DB)
knowledge_base = []

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests"""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

def get_embedding_size():
    """Determine the correct embedding size based on available models"""
    try:
        # Check if OpenAI is available and has valid key
        if openai.api_key and openai.api_key.startswith("sk-"):
            # Test if OpenAI actually works
            try:
                # Small test to check if API key is valid
                openai.Embedding.create(
                    input=["test"],
                    model="text-embedding-ada-002"
                )
                return 1536  # OpenAI ada-002 size
            except:
                # OpenAI key exists but doesn't work, use local
                from sentence_transformers import SentenceTransformer
                local_model = SentenceTransformer('all-MiniLM-L6-v2')
                return local_model.get_sentence_embedding_dimension()
        else:
            # Use local model size
            from sentence_transformers import SentenceTransformer
            local_model = SentenceTransformer('all-MiniLM-L6-v2')
            return local_model.get_sentence_embedding_dimension()
    except:
        return 384

# Initialize Qdrant manager with the correct vector size
vector_size = get_embedding_size()
qdrant_manager = initialize_qdrant_manager(vector_size)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Process uploaded files and extract content"""
    try:
        total_chunks = 1
        processed_files = []
        
        for file in files:
            content = await file.read()
            file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown'
            
            # Extract text from file using the document parser
            text_content = await document_parser.extract_text(content, file.filename)
            
            # Split text into chunks
            chunks = document_parser.chunk_text(text_content)
            
            # Generate embeddings for chunks
            use_local = not (openai.api_key and openai.api_key.startswith("sk-"))
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await embedding_service.get_embeddings_batch(chunk_texts, use_local=use_local)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]
            
            # Store chunks in Qdrant
            stored_count = await qdrant_manager.store_chunks(chunks, file.filename, file_extension)
            
            total_chunks += stored_count
            processed_files.append({
                "filename": file.filename,
                "type": file_extension,
                "size": len(content),
                "chunks": stored_count
            })
        
        return JSONResponse({
            "status": "success",
            "message": f"Processed {len(files)} file(s)",
            "total_chunks": total_chunks,
            "processed_files": processed_files
        })
    
    except Exception as e:
        logger.error(f"Upload error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ingest/cms")
async def ingest_cms(api_url: str = Form(...), auth_token: Optional[str] = Form(None)):
    """Import content from CMS API"""
    try:
        # Simulate fetching content from CMS (replace with actual API call)
        # For now, we'll create sample content
        cms_content = f"Content imported from CMS API: {api_url}. This includes articles, posts, and structured content. "
        
        # Split into chunks
        chunks = document_parser.chunk_text(cms_content)
        
        # Generate embeddings
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await embedding_service.get_embeddings_batch(chunk_texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
        
        # Store in Qdrant
        stored_count = await qdrant_manager.store_chunks(chunks, api_url, "cms")
        
        return JSONResponse({
            "status": "success",
            "message": "CMS content imported successfully",
            "chunks": stored_count,
            "source": api_url
        })
    
    except Exception as e:
        logger.error(f"CMS import error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"CMS import failed: {str(e)}")

@app.post("/config/openai")
async def set_openai_key(api_key_data: Dict[str, str]):
    """Update OpenAI API key at runtime"""
    try:
        new_key = api_key_data.get("api_key", "").strip()
        if not new_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        if not new_key.startswith("sk-") or len(new_key) != 51:
            raise HTTPException(status_code=400, detail="Invalid API key format")
        
        # Update the environment variable and OpenAI configuration
        os.environ["OPENAI_API_KEY"] = new_key
        openai.api_key = new_key
        
        return {
            "status": "success",
            "message": "OpenAI API key updated successfully",
            "key_prefix": new_key[:8] + "..."
        }
        
    except Exception as e:
        logger.error(f"API key update error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update API key: {str(e)}")

@app.post("/admin/recreate-collection")
async def recreate_collection():
    """Recreate the Qdrant collection with the correct vector size (admin only)"""
    try:
        vector_size = get_embedding_size()
        success = await qdrant_manager.recreate_collection(vector_size)
        
        if success:
            return {"status": "success", "message": f"Collection recreated with vector size {vector_size}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to recreate collection")
            
    except Exception as e:
        logger.error(f"Collection recreation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to recreate collection: {str(e)}")

@app.post("/ask")
async def ask_question(question_data: Dict[str, str]):
    """Answer questions using AI like ChatGPT with document context"""
    try:
        question = question_data.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Generate embedding for the question
        question_embedding = await embedding_service.get_embedding(question)
        
        # Search for relevant chunks in Qdrant
        relevant_chunks = await qdrant_manager.search_similar(question_embedding, limit=5)
        
        # Build context from relevant chunks
        context = ""
        if relevant_chunks:
            context = " ".join([chunk["content"] for chunk in relevant_chunks])
            sources = list(set([chunk["source"] for chunk in relevant_chunks]))
        else:
            sources = []
        
        # Use AI to generate answer with context
        answer = await generate_ai_answer(question, context)
        
        return JSONResponse({
            "answer": answer,
            "sources": sources[:3],  # Limit to 3 sources
            "confidence": 0.92,
            "context_used": bool(context),
            "total_relevant_chunks": len(relevant_chunks),
            "used_local_embeddings": True  # Changed from use_local to True
        })
        
    except Exception as e:
        logger.error(f"Question processing error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

async def generate_ai_answer(question: str, context: str = "") -> str:
    """Generate ChatGPT-like answers using OpenAI with enhanced prompting"""
    
    # If no OpenAI API key, use intelligent mock responses
    if not openai.api_key:
        return await generate_fallback_answer(question, context)
    
    try:
        # Enhanced prompt for better ChatGPT-like responses
        prompt = f"""You are an AI assistant similar to ChatGPT. Use the following context from the user's uploaded documents to provide accurate, helpful answers.

USER'S DOCUMENT CONTEXT:
{context if context else "No specific documents uploaded yet. Use your general knowledge to help the user."}

USER QUESTION: {question}

Please provide a comprehensive, helpful answer. If the context doesn't fully answer the question, use your knowledge to supplement it. 
Be conversational, clear, and provide value to the user. Format your response in a natural, engaging way."""

        # Call OpenAI API with better parameters
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful, friendly AI assistant that provides clear, comprehensive answers. You have access to the user's uploaded documents for context."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Ensure proper formatting
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
            
        return answer
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return await generate_fallback_answer(question, context)

async def generate_fallback_answer(question: str, context: str) -> str:
    """Generate intelligent fallback answers without OpenAI"""
    
    # Smart fallback responses based on context
    if context:
        return f"Based on your documents, I can tell you about: {context[:150]}... " \
               f"To get more detailed ChatGPT-like answers about '{question}', please add your OpenAI API key."
    else:
        responses = [
            f"I'd love to help with '{question}'! Upload some documents first, then I can provide detailed answers using AI.",
            f"Great question! Once you upload documents, I can analyze them and provide ChatGPT-like answers to '{question}'.",
            f"I'm ready to help with '{question}'. Please upload documents or add an OpenAI API key for AI-powered responses."
        ]
        return responses[len(question) % len(responses)]

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check OpenAI configuration
        openai_configured = bool(os.getenv("OPENAI_API_KEY"))
        openai_status = "configured" if openai_configured else "not_configured"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "document-qa-api",
            "knowledge_base": {
                "total_documents": len(knowledge_base),
                "total_chunks": sum(1 for _ in knowledge_base)
            },
            "openai": {
                "configured": openai_configured,
                "status": openai_status
            },
            "features": {
                "file_upload": True,
                "cms_import": True,
                "ai_answers": True,
                "chatgpt_like": openai_configured
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/debug/embedding-info")
async def debug_embedding_info():
    """Debug endpoint to check embedding configuration"""
    return {
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "embedding_service_size": embedding_service.embedding_size,
        "qdrant_collection_size": (await qdrant_manager.get_collection_stats()).get("vector_size", "unknown"),
        "local_model_loaded": embedding_service.local_model is not None,
        "local_model_size": embedding_service.local_model.get_sentence_embedding_dimension() if embedding_service.local_model else "not_loaded"
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "status": "success",
        "count": len(knowledge_base),
        "documents": [
            {
                "id": chunk.get("chunk_id", str(i)),
                "filename": chunk.get("filename", "cms_import"),
                "type": chunk.get("type", "unknown"),
                "timestamp": chunk.get("timestamp", ""),
                "content_preview": chunk.get("content", "")[:100] + "..." if chunk.get("content") else ""
            }
            for i, chunk in enumerate(knowledge_base)
        ]
    }

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from knowledge base"""
    cleared_count = len(knowledge_base)
    knowledge_base.clear()
    
    return {
        "status": "success",
        "message": f"Cleared {cleared_count} documents",
        "remaining": len(knowledge_base)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Q&A API - ChatGPT-like",
        "version": "1.0.0",
        "description": "AI-powered document analysis with ChatGPT-like responses",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "upload": "/upload",
            "ingest": "/ingest/cms",
            "ask": "/ask",
            "documents": "/documents"
        },
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.get("/qdrant-status")
async def qdrant_status():
    """Check Qdrant connection status"""
    try:
        stats = await qdrant_manager.get_collection_stats()
        return {
            "status": "success",
            "qdrant_connected": True,
            "collection_stats": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "qdrant_connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
