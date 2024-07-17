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
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
)
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
        self.store = {}
        mem = psutil.virtual_memory()
        available_memory_gb = mem.available / (1024 ** 3)
        logging.info(f"Initializing ChromaManager - Available memory: {available_memory_gb:.2f} GB")

    def load_model(self, model_name="llama3:8b"):
        try:
            logging.info("Loading embedding model...")

            mem_before = psutil.virtual_memory()
            available_memory_before_gb = mem_before.available / (1024 ** 3)
            logging.info(f"Memory before loading model: {available_memory_before_gb:.2f} GB")

            self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
            self.llm = ChatOllama(model=model_name)
            print(f"run {model_name}")

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

    def create_custom_stuff_documents_chain(self,llm, prompt_temp):
        def format_docs(inputs: dict) -> str:
            return DEFAULT_DOCUMENT_SEPARATOR.join(
                format_document(doc) for doc in inputs[DOCUMENTS_KEY]
            )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_temp),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        return (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(x)
            ).with_config(run_name="format_inputs")
            | qa_prompt
            | llm
            | StrOutputParser()
        ).with_config(run_name="custom_stuff_documents_chain")

    def build_chain(self, k, prompt_temp):
        self.chroma_ret = self.chroma_db.as_retriever(search_kwargs={"k": k})
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.chroma_ret, contextualize_q_prompt
        )
        #question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        question_answer_chain = self.create_custom_stuff_documents_chain(self.llm, prompt_temp)
        
        qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        self.rag_chain = RunnableWithMessageHistory(
            qa_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def invoke(self, query):
        res = self.rag_chain.invoke({"input": query},
            config={
                "configurable": {"session_id": "abc2333"}
            },
        )
        #print(res)
        return res["answer"]

#helper for car_stats
def format_document(doc):
    page_content = doc.page_content
    car_stats = doc.metadata.get('car_stats', 'No car stats available')
    return f"{page_content}\n Car stats Related: {car_stats}"

def main():
    config_path = "../config/config.yaml"
    config = load_config(config_path)

    manager = ChromaManager(config, 'lotus')
    manager.load_model("gemma2:9b")
    manager.check_db()
    prompt_template = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    {context}"""

    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )
    manager.build_chain(5, prompt_template)
    while True:
        test_prompt = input("Please enter your question: ")
        print("Answer")
        print(manager.invoke(test_prompt))
        #print("")
    # result = manager.retrieve_top_k(test_prompt)
    # print(result)


if __name__ == "__main__":
    main()
