from vector_search.vector_stores import get_ollama_vector_store

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


__invoke_agent("what is self reflection?")