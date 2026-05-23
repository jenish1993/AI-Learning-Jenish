"""this is going to create vector store out of provided pdf files."""

from pdf_reader import read_pdf_file
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def __split_documents(docs: list[Document]):
    #splitting/chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000, # here comes chunking and fine tuning
        chunk_overlap= 200, # decides how many chuks/words will overlap over the diffrent chunks.
        add_start_index = True
    )

    #split/chunk documents 
    chunks = text_splitter.split_documents(docs)

    return chunks

def get_vector_store(file_paths: list[str], store_name: str, persist_directory = "./chroma_db/db_file"):
    # #check if file path is available.
    # if len(file_paths) is 0:
    #     raise Exception("No File Paths provided")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=store_name,
        embedding_function = embeddings,
        persist_directory = persist_directory
    )

    #read all files
    for path in file_paths:
        doc_splits = __split_documents(read_pdf_file(file_path=path))
        #add all documents into vector store.
        vector_store.add_documents(doc_splits)

    return vector_store




#test 
#this can be replaced with multiple file to create single vector store out of multiple pdf files.
file_path = ["docs/financial-results-qr-mar-31-2026.pdf"]
query = "What is TCS AI Investment for Q2 2026?"

vs = get_vector_store(file_paths=file_path, store_name= "tcs_q2_2026")

retriver = vs.as_retriever(
    search_type="similarity",
    search_kwargs ={"k": 1}
)

query_response = retriver.batch([query])

print(query_response)