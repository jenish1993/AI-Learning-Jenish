from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )

    all_doc_splits = text_splitter.split_documents(docs)

    return all_doc_splits