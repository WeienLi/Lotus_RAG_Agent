import sys
import os
import json
import logging
import yaml
import re
from typing import List
from tqdm import tqdm

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        lines = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines]
        return lines


class ChromaManager:
    def __init__(self, config, collection_name, batch_size=5):
        self.file_path = config['file_path']
        self.persist_directory = config['persist_directory']
        self.embeddings_model_name = config['embeddings_model_name']
        self.llm = ChatOllama(model=config['llm_s'])
        self.collection_name = collection_name
        self.batch_size = batch_size

        logging.info("Initializing ChromaManager")

    def load_model(self):
        try:
            logging.info("Loading embedding model...")

            self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)

            self.chroma_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                relevance_score_fn="l2"  # l2, ip, cosine
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
                "id": str(content_dict.get("id"))
            }
            metadata_list.append(metadata)

        for i in tqdm(range(0, len(content_list), self.batch_size), desc="Storing database"):
            batch_contents = content_list[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            embeddings_list = self.embeddings.embed_documents(batch_contents)
            for content, embedding, metadata in zip(batch_contents, embeddings_list, batch_metadata):
                self.chroma_db.add_texts([content], embeddings=[embedding], metadatas=[metadata], ids=[metadata['id']])

        self.chroma_db.persist()
        logging.info("Chroma vector database created and persisted successfully.")

    def insert(self, documents, metadatas):
        embeddings_list = self.embeddings.embed_documents(documents)
        for content, embedding, metadata, doc_id in zip(documents, embeddings_list, metadatas):
            self.chroma_db.add_texts([content], embeddings=[embedding], metadatas=[metadata], ids=[metadata['id']])
        self.chroma_db.persist()
        logging.info("New documents inserted successfully.")

    def check_db(self):
        logging.info(f"There are {self.chroma_db._collection.count()} in the collection")

    def retrieve_top_k(self, prompt, k=5):
        return self.chroma_db.similarity_search_with_score(prompt, k=k)

    def get_db_as_ret(self,search_type = "similarity", search_kwargs = {"k": 5}):
        return self.chroma_db.as_retriever(search_type = search_type, search_kwargs = search_kwargs)

    def multi_query_retrieve_top_k(self, prompt, k=5):
        output_parser = LineListOutputParser()

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
                        different versions of the given user question to retrieve relevant documents from a vector 
                        database. By generating multiple perspectives on the user question, your goal is to help
                        the user overcome some of the limitations of the distance-based similarity search. 
                        Provide these alternative questions separated by newlines.
                        Original question: {question}""",
        )

        llm_chain = QUERY_PROMPT | self.llm | output_parser
        queries = llm_chain.invoke(prompt)

        results = []
        unique_ids = []
        for query in queries:
            if len(query) < 1:
                continue
            for result, score in self.chroma_db.similarity_search_with_score(query, k=k):
                if result.metadata['id'] not in unique_ids:
                    unique_ids.append(result.metadata['id'])
                    results.append((result, score))

        results.sort(key=lambda x: x[1])
        return results
    
    def evaluate_retrieval(self, queries, top_k=5):
        total_queries = len(queries)
        score = 0.0
        id_match_num = 0
        page_match_num = 0
        for query in tqdm(queries):
            question = query["question"]
            expected_page_num = query["page_num"]
            expected_id = query["id"]

            results = self.retrieve_top_k(question, k=top_k)
            # results = self.multi_query_retrieve_top_k(question, k=top_k)
            id_found = False
            page_num_found = False
            for result, _ in results:
                if str(result.metadata['id']) == str(expected_id):
                    id_found = True
                    score += 1
                    id_match_num += 1
                    page_match_num += 1
                    break
                elif str(result.metadata['page_number']) == str(expected_page_num):
                    page_num_found = True

            if not id_found and page_num_found:
                score += 1
                page_match_num += 1

        accuracy = score / total_queries

        logging.info(
            f"Total Query: {total_queries}, "
            f"Top {top_k} Accuracy: {accuracy * 100:.2f}%, "
            f"id Match Num: {id_match_num}, "
            f"Page Match Num: {page_match_num}"
        )

        return accuracy


def main():
    config_path = "../config/config.yaml"
    config = load_config(config_path)

    manager = ChromaManager(config, 'lotus')
    manager.load_model()
    # manager.load_and_store_data()
    # manager.check_db()

    # test_prompt = "What is Mark"
    # result = manager.retrieve_top_k(test_prompt)
    # print(result)

    test_file_path = '../Data/test_data/rag_test_part1.json'
    with open(test_file_path, 'r') as file:
        test_queries = json.load(file)

    accuracy = manager.evaluate_retrieval(test_queries, 10)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()