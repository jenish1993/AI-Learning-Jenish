from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from .documents_loader.web_doc_loader import load_web_page
from .text_splitter import split_document_text
import bs4

# Ollama Stuff

def get_ollama_vector_store(page_url:str):

    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = InMemoryVectorStore(embedding=embeddings)

    #load document
    bs_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    docs = load_web_page(page_url=page_url, 
                    bs_kwargs={
                        "parse_only": bs_strainer
                    })
    
    #split document
    splitted_docs = split_document_text(docs)

    #add document into vector store.
    vector_store.add_documents(documents=splitted_docs)

    return vector_store


# End Ollama Stuff