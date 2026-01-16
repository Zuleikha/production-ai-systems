import PyPDF2
import io
from typing import Optional

class PDFParser:
    """Parse PDF documents and extract text."""
    
    def extract_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            raise ValueError(f"PDF parsing failed: {str(e)}")
