import sys
import os
import json
import logging
import psutil
import yaml
from tqdm import tqdm
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class ChromaManager:
    def __init__(self, config, collection_name, batch_size=5):
        self.file_path = config['file_path']
        self.persist_directory = config['persist_directory']
        self.embeddings_model_name = config['embeddings_model_name']
        self.collection_name = collection_name
        self.batch_size = batch_size

        mem = psutil.virtual_memory()
        available_memory_gb = mem.available / (1024 ** 3)
        logging.info(f"Initializing ChromaManager - Available memory: {available_memory_gb:.2f} GB")

    def load_model(self):
        try:
            logging.info("Loading embedding model...")

            mem_before = psutil.virtual_memory()
            available_memory_before_gb = mem_before.available / (1024 ** 3)
            logging.info(f"Memory before loading model: {available_memory_before_gb:.2f} GB")

            self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)

            mem_after = psutil.virtual_memory()
            available_memory_after_gb = mem_after.available / (1024 ** 3)
            logging.info(f"Memory after loading model: {available_memory_after_gb:.2f} GB")

            memory_used_gb = available_memory_before_gb - available_memory_after_gb
            logging.info(f"Memory used by model: {memory_used_gb:.2f} GB")

            self.chroma_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def load_and_store_data(self):
        loader = JSONLoader(file_path=self.file_path, jq_schema=".[]", text_content=False)
        documents = loader.load()

        content_list = []
        metadata_list = []

        for doc in documents:
            content_dict = json.loads(doc.page_content)
            content = content_dict.get("content", "")
            content_list.append(content)
            metadata = {
                "page_number": content_dict.get("page_number"),
                "car_stats": json.dumps(content_dict.get("car_stats", "N/A")) if isinstance(
                    content_dict.get("car_stats"), list) else content_dict.get("car_stats", "N/A"),
            }
            metadata_list.append(metadata)

        for i in tqdm(range(0, len(content_list), self.batch_size), desc="Storing database"):
            batch_contents = content_list[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            embeddings_list = self.embeddings.embed_documents(batch_contents)
            for content, embedding, metadata in zip(batch_contents, embeddings_list, batch_metadata):
                self.chroma_db.add_texts([content], embeddings=[embedding], metadatas=[metadata])

        self.chroma_db.persist()
        logging.info("Chroma vector database created and persisted successfully.")

    def check_db(self):
        logging.info(f"There are {self.chroma_db._collection.count()} in the collection")

    def retrieve_top_k(self, prompt, k=5):
        return self.chroma_db.similarity_search_with_score(prompt, k=k)


def main():
    config_path = "../config/config.yaml"
    config = load_config(config_path)

    manager = ChromaManager(config, 'lotus')
    manager.load_model()
    manager.load_and_store_data()
    manager.check_db()

    test_prompt = "What is Mark"
    result = manager.retrieve_top_k(test_prompt)
    print(result)


if __name__ == "__main__":
    main()
