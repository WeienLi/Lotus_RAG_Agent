import sys
import os
import json
import logging
import re
from typing import List
from tqdm import tqdm

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain_core.output_parsers import BaseOutputParser

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        self.llm = ChatOllama(model=config['llm'])
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

    def get_retriever(self, k=5, retriever_type="chroma"):
        chroma_docs = self.chroma_db.get()
        documents = [Document(page_content=doc, metadata=metadata)
                     for doc, metadata in zip(chroma_docs['documents'], chroma_docs['metadatas'])]
        if retriever_type == "ensemble":
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = k

            faiss_vectorstore = FAISS.from_documents(documents, self.embeddings)
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

            chroma_retriever = self.chroma_db.as_retriever(search_kwargs={"k": k})

            retrievers = [bm25_retriever, faiss_retriever, chroma_retriever]

            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=[1 / len(retrievers) for i in range(len(retrievers))]
            )
            return ensemble_retriever

        if retriever_type == "faiss":
            faiss_vectorstore = FAISS.from_documents(documents, self.embeddings)
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})
            return faiss_retriever

        else:
            retriever = self.chroma_db.as_retriever(search_kwargs={"k": k})
            return retriever

    def evaluate_retrieval(self, queries, top_k=5, retriever_type="chroma", save=False):
        total_queries = len(queries)
        score = 0.0
        id_match_num = 0
        page_match_num = 0
        results_list = []
        retriever = self.get_retriever(top_k, retriever_type)

        for query in tqdm(queries):
            question = query["question"]
            expected_page_num = query["page_num"]
            expected_id = query["id"]

            results = retriever.invoke(question)
            id_found = False
            page_num_found = False
            for result in results:
                if isinstance(result, tuple):
                    result = result[0]
                if str(result.metadata['id']) == str(expected_id):
                    id_found = True
                    page_num_found = True
                    score += 1
                    id_match_num += 1
                    page_match_num += 1
                    break
                elif str(result.metadata['page_number']) == str(expected_page_num):
                    page_num_found = True

            if not id_found and page_num_found:
                # score += 1
                page_match_num += 1

            if save:
                result_entry = {
                    'id_found': id_found,
                    'expected_id': expected_id,
                    'page_num_found': page_num_found,
                    'expected_page_num': expected_page_num,
                    'results': [{'metadata': result.metadata, 'content': result.page_content, 'distance': score} for
                                result, score in results]
                }
                results_list.append(result_entry)

        accuracy = score / total_queries

        logging.info(
            f"Retriever: {retriever_type}, "
            f"Total Query: {total_queries}, "
            f"Top {top_k} Accuracy: {accuracy * 100:.2f}%, "
            f"id Match Num: {id_match_num}, "
            f"Page Match Num: {page_match_num}"
        )

        if save:
            with open('result.json', 'w') as f:
                json.dump(results_list, f, indent=4)

        return accuracy

    def get_db_as_ret(self, search_type="similarity", search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.chroma_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
