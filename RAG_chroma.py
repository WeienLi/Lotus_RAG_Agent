import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
load_dotenv()

# Initialize the HuggingFace embeddings model
embeddings_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Define persist directory and collection name
persist_directory = "/Users/barryli/Desktop/Fuck/chroma/test"
collection_name = "test"

chroma_db = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)

def check_db():
    print("There are", chroma_db._collection.count(), "in the collection")

# Step 2: Create RAG test framework to retrieve top-k similar contents
def retrieve_top_k(prompt, k=5):
    #retriever = chroma_db.as_retriever()
    #results = retriever.get_relevant_documents(prompt, limit=k)
    #return results
    return chroma_db.similarity_search_with_score(prompt, k=k)


"""
This is not correct
def generate_with_llama3(prompt, retrieved_docs):
    
    model = Ollama(model="llama3", callbacks=[StreamingStdOutCallbackHandler()])
    context = "\n\n".join(doc["content"] for doc in retrieved_docs)
    full_prompt = f"{prompt}\n\n{context}"
    
    response = model.generate(full_prompt)
    return response
"""

def main():
    check_db()

    test_prompt = "What is Mark"
    result = retrieve_top_k(test_prompt)
    for res in result:
        print("MetaData")
        print(res[0].metadata)
        print("Conf")
        print(res[1])
        print("Page_Content")
        print(res[0].page_content)
        print("-------------" * 5)
    """
    # Retrieve top-k similar contents
    k = 5
    retrieved_docs = retrieve_top_k(test_prompt, k=k)
    print("Retrieved documents:", json.dumps(retrieved_docs, indent=2))

    # Flag to switch between RAG testing and LLAMA3 generation
    use_llama3 = False  # Set to True to use LLAMA3 for generation

    if use_llama3:
        # Generate output with LLAMA3
        output = generate_with_llama3(test_prompt, retrieved_docs)
        print("Generated output with LLAMA3:", output)
    else:
        # Pure RAG testing
        print("Top-k retrieved contents for RAG testing:")
        for doc in retrieved_docs:
            print(json.dumps(doc, indent=2))
    """
if __name__ == "__main__":
    main()
