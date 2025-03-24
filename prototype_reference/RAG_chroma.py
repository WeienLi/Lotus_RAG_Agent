import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import yaml


"""
RAG_chroma.py

This is a prototype testing file that is first used to test the RAG functionality using ChromaDB
"""

def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def check_db(db):
    """Check number of entries in the current database

    Args:
        db (Chroma): The vector database that is intialized
    """
    print("There are", db._collection.count(), "in the collection")

def retrieve_top_k(db, prompt, k=5):
    """Retrieve Top K most similar Documents

    Args:
        db (Chroma): The vector database that we want to retrieve from
        prompt (str): Prompt of the question
        k (int, optional): K cloest documents that we want to retrieve. Defaults to 5.

    Returns:
        List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
    """
    #retriever = chroma_db.as_retriever()
    #results = retriever.get_relevant_documents(prompt, limit=k)
    #return results
    return db.similarity_search_with_score(prompt, k=k)



def main():
    """Main Function to test the chromaDB retireval
    """
    load_dotenv()
    config_path = "./prototype_reference/config/config.yaml"
    config = load_config(config_path)
    
    embeddings_model_name = config["embeddings_model_name"]
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    persist_directory = config["persist_directory"]
    collection_name = config["collection_name"]

    #Intialize the chroma_database instance and check the number of records stored
    chroma_db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    check_db(chroma_db)

    # Test the Prompt and the retrieving result
    test_prompt = "What is Mark"
    result = retrieve_top_k(chroma_db, test_prompt)
    for res in result:
        print("MetaData")
        print(res[0].metadata)
        print("Conf")
        print(res[1])
        print("Page_Content")
        print(res[0].page_content)
        print("-------------" * 5)
        
if __name__ == "__main__":
    main()
