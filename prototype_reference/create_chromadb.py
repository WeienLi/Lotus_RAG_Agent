import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
import chromadb
import json
from langchain_huggingface import HuggingFaceEmbeddings
import yaml

"""
create_chromadb.py

This is a prototype testing file that is first used to create ChromaDB and add data to it.
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

def load_documents(file_path):
    """load documents from JSON file

    Args:
        file_path (str): path towards the JSON file we want to load

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a document from the JSON file.
    """
    loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
    documents = loader.load()
    return documents


def intialize_chroma(persist_directory,collection_name,model):
    """_summary_

    Args:
        persist_directory (str): Path of a directory where the chromaDB is kept
        collection_name (str): Collection name that we want to use
        model (Embedding Model): The function that we use to embedded the data

    Returns:
        Chroma: Instance of Chroma Database that is intialized
    """
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    collection = persistent_client.get_or_create_collection(collection_name)
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=model,
        persist_directory=persist_directory
    )
    print(f"Persist directory: {persist_directory}")
    return langchain_chroma

def add_and_persist(langchain_chroma, documents,model):
    """Add and persist the chroma database

    Args:
        langchain_chroma (Chroma): Instance of Chroma Database that is intialized
        documents (list[dict]): A list of dictionaries where each dictionary represents a document from the JSON file.
        model (Embedding Model): The function that we use to embedded the data
    """
    content_list = []
    metadata_list = []

    for doc in documents:
        content_dict = json.loads(doc.page_content)
        content = content_dict.get("content", "")
        content_list.append(content)
        # Parse the metadata by adding pagenumber as well as car_stats if it exists into the metadata
        metadata = {
            "page_number": content_dict.get("page_number"),
            "car_stats": json.dumps(content_dict.get("car_stats", "N/A")) if isinstance(content_dict.get("car_stats"),
                                                                                        list) else content_dict.get(
                "car_stats", "N/A"),
        }
        metadata_list.append(metadata)

    #Embedd the content and add them to chroma_db
    embeddings_list = model.embed_documents(content_list)
    for content, embedding, metadata in zip(content_list, embeddings_list, metadata_list):
        langchain_chroma.add_texts([content], embeddings=[embedding], metadatas=[metadata])

    langchain_chroma.persist()
    print("Chroma vector database created and persisted successfully.")

def main():
    load_dotenv()
    config_path = "./prototype_reference/config/config.yaml"
    config = load_config(config_path)
    file_path = config["data_path"]
    collection_name = config["collection_name"]
    documents = load_documents(file_path)
    model = HuggingFaceEmbeddings(model_name=config["embeddings_model_name"])
    persist_directory = config["persist_directory"]
    langchain_chroma = intialize_chroma(persist_directory,collection_name, model)
    add_and_persist(langchain_chroma, documents, model)
    
if __name__ == '__main__':
    main()