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
    """
    A class to prase text output into a list of clean strings, removing numerical prefixes

    Inherit From:
        Langchain's base parser class for handling structured output
        
    Methods:
        parse(self, text: str):
            Parses the input text, removes numerical prefixes and return them as a list of strings
    """
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        lines = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines]
        return lines


class ChromaManager:
    """
    A class use to manage the Chroma Database for storing, retrieving as well as evluating text embeddings
    
    Attributes:
        persist_directory (str): Directory where Chroma database is persisted
        embeddings_model_name (str): Name of the embedding model for generating embedding
        collection_name (str): Name of the Chroma database collection
        batch_size (int): Batch size for inserting to database
    
    Methods:
        create_collection(collection_name=None):
            Intialize the chroma database collection
            
        load_and_store_data(dir_path, ignore_range=False):
            Loads JSON documents and stores its content in Chroma database
        
        insert(documents, metadatas):
            Inserts new documents into current Chroma database
            
        remove_all():
            Deletes all documents in the Chroma collection
        
        check_db():
            Check the number of documents within the current database
        
        get_retriever(k=5, retriever_type="chroma"):
            Returns a retriever based on retrieval strategy
        
        evaluate_retrieval(queries, top_k=5, retriever_type="chroma", save=False):
            Evaluates retrieval accuracy using a set of queries
            
    """
    def __init__(self, config, collection_name):
        """
        Initializes a ChromaManager instance with configuration settings

        Args:
            config (dict): dictionary containing configuration startegy
            collection_name (str): Name of the Chroma database collection
        """
        self.persist_directory = config['persist_directory']
        self.embeddings_model_name = config['embeddings_model_name']
        #self.llm = ChatOllama(model=config['llm'])
        self.collection_name = collection_name
        self.batch_size = 5
        try:
            logging.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
            self.chroma_db = None
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")

    def create_collection(self, collection_name=None):
        """
        Intialize the chroma database collection

        Args:
            collection_name (str, optional): Name of the Chroma database collection Defaults to None
        """
        if collection_name is not None:
            self.collection_name = collection_name

        logging.info(f"Initializing ChromaManager collection {self.collection_name}")

        self.chroma_db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            relevance_score_fn="l2"  # l2, ip, cosine
        )

    def load_and_store_data(self, dir_path, ignore_range=False):
        """
        Loads JSON documents and stores its content in Chroma database

        Args:
            dir_path (str): Path to the directory containing JSON files we want to load
            ignore_range (bool, optional): If True, ignores page range constraints Defaults to False
        """
        content_list = []
        metadata_list = []
        global_id = 0

        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                json_file = os.path.join(dir_path, filename)
                loader = JSONLoader(file_path=json_file, jq_schema=".[]", text_content=False)
                documents = loader.load()

                page_range = json.loads(documents[0].page_content)
                page_start = page_range['start']
                page_end = page_range['end']

                for doc in documents[1:]:
                    content_dict = json.loads(doc.page_content)
                    content = content_dict.get("content", "")
                    page_number = content_dict.get("page_number")
                    if int(page_start) <= int(page_number) <= int(page_end) or ignore_range:
                        content_list.append(content)
                        metadata = {
                            "filename": filename,
                            "page_number": page_number,
                            "car_stats": json.dumps(content_dict.get("car_stats", "")) if isinstance(
                                content_dict.get("car_stats"), list) else content_dict.get("car_stats", ""),
                            "id": str(content_dict.get("id")),
                            "global_id": global_id
                        }
                        metadata_list.append(metadata)
                        global_id += 1

        for i in tqdm(range(0, len(content_list), self.batch_size), desc="Storing database"):
            batch_contents = content_list[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            embeddings_list = self.embeddings.embed_documents(batch_contents)
            for content, embedding, metadata in zip(batch_contents, embeddings_list, batch_metadata):
                self.chroma_db.add_texts([content], embeddings=[embedding], metadatas=[metadata])

        logging.info("Chroma vector database created and persisted successfully.")
        logging.info(f"There are {self.chroma_db._collection.count()} in the {self.collection_name} collection")

    def insert(self, documents, metadatas):
        """
        Inserts new documents into current Chroma database

        Args:
            documents (List[str]): Documents we want to add into the Chroma database
            metadatas (List[dict]): Document's metadata
        """
        embeddings_list = self.embeddings.embed_documents(documents)
        for content, embedding, metadata, doc_id in zip(documents, embeddings_list, metadatas):
            self.chroma_db.add_texts([content], embeddings=[embedding], metadatas=[metadata], ids=[metadata['id']])

        logging.info("New documents inserted successfully.")

    def remove_all(self):
        """
        Deletes all documents in the Chroma collection
        """
        try:
            self.chroma_db.delete_collection()
            logging.info(f"Successful to remove documents from the collection '{self.collection_name}'")
        except Exception as e:
            logging.error(f"Failed to remove documents from the collection '{self.collection_name}': {e}")

    def check_db(self):
        """
        Check the number of documents within the current database
        """
        logging.info(f"There are {self.chroma_db._collection.count()} in the {self.collection_name} collection")

    def get_retriever(self, k=5, retriever_type="chroma"):
        """
        Returns a retriever based on retrieval strategy

        Args:
            k (int, optional): Number closest documents we want to retrieve Defaults to 5
            retriever_type (str, optional): retriever strategy we want to use Defaults to "chroma"

        Returns:
            BaseRetriever: Selected Retriever
        """
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
        """
        Evaluates the retrieval accuracy of the retriever strategy.
        
        Args:
            queries (list[dict]): Query we want to use to retrieve alongside the right page_number, id and right file
            top_k (int, optional): Number closest documents we want to retrieve Defaults to 5 Defaults to 5
            retriever_type (str, optional): retriever strategy we want to use Defaults to "chroma"
            save (bool, optional): Options on whether we want to save the results in json format Defaults to False

        Returns:
            float: The retrieval accuracy as a ratio of correctly retrieved IDs divided by total queries
        """
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
            expected_file = "car_stats.json"

            results = retriever.invoke(question)
            id_found = False
            page_num_found = False
            for result in results:
                if isinstance(result, tuple):
                    result = result[0]
                if str(result.metadata['filename']) != expected_file:
                    continue
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
                    'expected_file': expected_file,
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
