"""
Document processing module for extracting and cleaning text from PDF files.
"""

import os
import re
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles PDF text extraction and text cleaning."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize the extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Cleaned text to chunk
            
        Returns:
            List of dictionaries containing chunked text with metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                'id': f'chunk_{i}',
                'text': chunk,
                'chunk_index': i,
                'source': 'HR-Policy.pdf'
            })
        
        return chunked_documents
    
    def process_document(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Complete document processing pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of processed document chunks
        """
        print(f"Processing document: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(raw_text)} characters")
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        print(f"Cleaned text length: {len(cleaned_text)} characters")
        
        # Chunk text
        chunks = self.chunk_text(cleaned_text)
        print(f"Created {len(chunks)} chunks")
        
        return chunks


if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    pdf_path = "HR-Policy (1).pdf"
    
    if os.path.exists(pdf_path):
        chunks = processor.process_document(pdf_path)
        print(f"\nFirst chunk preview:")
        print(chunks[0]['text'][:200] + "...")
    else:
        print(f"PDF file not found: {pdf_path}")

