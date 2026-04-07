import io
from typing import Optional

import PyPDF2


class PDFParser:
    """Extract plain text from PDF documents."""

    def extract_text(self, pdf_content: bytes) -> str:
        """Return concatenated text from all pages of a PDF.

        Args:
            pdf_content: Raw PDF bytes.

        Returns:
            Stripped plain-text string.

        Raises:
            ValueError: If the PDF cannot be parsed.
        """
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()
        except Exception as e:
            raise ValueError(f"PDF parsing failed: {e}") from e
