import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_search import get_ollama_vector_store
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

def __vector_search(query: str):
    vector_store = get_ollama_vector_store(URL)

    # make similarity search.
    retrieved_docs = vector_store.similarity_search(query, k=2)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata},\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

def __invoke_agent(prompt: str):
    ser, ret_docs = __vector_search(query=prompt)

    llm = ChatOllama(model="llama3", temperature=0.5)

    # 1. Define the role, inject context, and pass the user's prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question:\n\n{context}"),
        ("human", "{question}")
    ])

    # 2. Chain the prompt template and the LLM together
    chain = prompt_template | llm
    
    # 3. Execute/Invoke with the variables mapped
    response = chain.invoke({"context": ser, "question": prompt})

    # Print the text content of the response
    print(response.content)


__invoke_agent("what is self reflection?")