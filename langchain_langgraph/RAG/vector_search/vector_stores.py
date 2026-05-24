from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from documents_loader.web_doc_loader import load_web_page
from text_splitter import split_document_text
import bs4

# Ollama Stuff
#
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

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
def __vector_serach(query: str):
    vector_store = get_ollama_vector_store(URL)

    # make similarity search.
    retrived_docs = vector_store.similarity_search(query, k=2)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata},\nContent: {doc.page_content}")
        for doc in retrived_docs
    )

    return serialized, retrived_docs

def __invoke_agent(prompt: str):
    ser, ret_docs = __vector_serach(query=prompt)

    print(ser)

__invoke_agent("what is Locality-Sensitive Hashing?")
