import PyPDF2
import docx2txt
import pandas as pd
from io import BytesIO, StringIO
from fastapi import HTTPException
from typing import Dict, Any
import aiofiles
import asyncio
import logging
from pdfminer.high_level import extract_text as pdfminer_extract_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    """Handles text extraction from various file formats with improved PDF handling"""
    
    @staticmethod
    async def extract_text(file_content: bytes, filename: str) -> str:
        """Extract text from various file formats"""
        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
        
        try:
            if file_extension == 'pdf':
                return await DocumentParser._extract_pdf(file_content)
            elif file_extension in ['docx', 'doc']:
                return await DocumentParser._extract_docx(file_content)
            elif file_extension == 'csv':
                return await DocumentParser._extract_csv(file_content)
            elif file_extension in ['xlsx', 'xls']:
                return await DocumentParser._extract_excel(file_content)
            elif file_extension == 'txt':
                return file_content.decode('utf-8')
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

            logger.info(f"Successfully extracted {len(result)} characters from {filename}")
            return result
   
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return f"Content from {filename}: Unable to extract full text. Error: {str(e)}"    
    @staticmethod
    async def _extract_pdf(content: bytes) -> str:
        """Extract text from PDF using multiple methods for better accuracy"""
        text = ""
        
        # Method 1: Try pdfminer (better for complex PDFs)
        try:
            text = pdfminer_extract_text(BytesIO(content))
            if text.strip():
                logger.info("PDF extracted successfully with pdfminer")
                return text
        except Exception as e:
            logger.warning(f"pdfminer failed: {e}")
        
        # Method 2: Fallback to PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info("PDF extracted with PyPDF2 fallback")
        except Exception as e:
            logger.error(f"PyPDF2 also failed: {e}")
            raise HTTPException(status_code=400, detail="PDF extraction failed")
        
        return text
    
    @staticmethod
    async def _extract_docx(content: bytes) -> str:
        """Extract text from DOCX"""
        return docx2txt.process(BytesIO(content))
    
    @staticmethod
    async def _extract_csv(content: bytes) -> str:
        """Extract text from CSV"""
        csv_text = content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_text))
        return df.to_string()
    
    @staticmethod
    async def _extract_excel(content: bytes) -> str:
        """Extract text from Excel files"""
        df = pd.read_excel(BytesIO(content))
        return df.to_string()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
        """Split text into overlapping chunks with context preservation"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to end at a sentence boundary
            if end < text_length:
                # Look for sentence endings near the chunk end
                sentence_end = text.rfind('. ', start + chunk_size - 100, end + 100)
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "content": chunk,
                    "start_index": start,
                    "end_index": end
                })
            
            start = max(start + chunk_size - overlap, end - overlap)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

# Singleton instance
document_parser = DocumentParser()
