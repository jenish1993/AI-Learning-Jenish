"""This File is going to help us read pdf files and create langchain documents from pdf for loading it on vector store."""
import pypdf
from langchain_core.documents import Document

def read_pdf_file(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content = page.extract_text() or "",
            metatadata = {
                "source": file_path,
                "page": i
            }
        ) for i, page in enumerate(reader.pages)
    ]


# #tests
# file_path = "docs/financial-results-qr-mar-31-2026.pdf"
# docs = read_pdf_file(file_path)
# print(len(docs))