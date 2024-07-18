import yaml
from functools import lru_cache
from abc import ABC, abstractmethod
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
)


def format_document(doc):
    page_content = doc.page_content
    car_stats = doc.metadata.get('car_stats', 'No car stats available')
    return f"{page_content}\n Car stats Related: {car_stats}"


class OllamaManager:
    def __init__(self, config, db_ret):
        self.llm = ChatOllama(model=config['llm'])
        self.qa_prompt = self._qa_template()
        self.gen_qa_prompt = self._gen_qa_template()
        self.store = {}
        self.db_ret = db_ret
        self.rag_chain = self.build_chain()

    @staticmethod
    def _history_template():
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
        return contextualize_q_prompt

    @staticmethod
    def _qa_template():
        # If you don't know the answer, just say that you don't know. \
        #                 Use Seven sentences maximum and keep the answer concise.\
        return """You are an assistant for question-answering tasks. \
                Use the following pieces of retrieved context to answer the question. \
                {context}"""

    @staticmethod
    def _gen_qa_template():
        return """You are an assistant for question-answering tasks. \
                Use three sentences maximum and keep the answer concise."""

    def custom_history_template(self, contextualize_q_system_prompt):
        self.history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def custom_qa_prompt(self, prompt):
        self.qa_prompt = prompt

    def change_model(self, new_model: str):
        self.llm = ChatOllama(model=new_model)

    def get_model_info(self):
        return self.llm.model_info()

    @staticmethod
    def create_custom_stuff_documents_chain(llm, prompt_temp):
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

    def if_query_rag(self, question, max_retry=3):
        prompt_template = """
        You are a smart assistant designed to categorize questions. You need to determine whether the user's question is a general daily question or a specific question that requires information from a specific dataset about car statistics. 

        The dataset includes detailed historical and technical data about various car models such as:
        - Model names and formulas (e.g., Mark I, Trials Car, Mark II, etc.)
        - Years of production
        - Number of examples built
        - Engine types and specifications (e.g., Austin Seven two-bearing side-valve, Ford 10 side-valve, etc.)
        - Dimensions (length, width, height)
        - Wheelbase
        - Weight

        Here are some example questions related to the dataset:
        - "What engine was used in the Mark I car?"
        - "How many Mark II cars were built?"
        - "Can you provide the specifications for the Mark VI?"
        - "What were the production years for the Mark VIII?"
        - "Please tell me something about lotus?"

        Any question that involves details about car models or mention about the keywords such as lotus, their specifications, history, or technical data should be categorized as requiring the specific dataset (Answer: YES).

        General daily questions might include:
        - "What's the weather like today?"
        - "How do I make a cup of coffee?"
        - "What's the capital of France?"
        - "What time is it?"

        For such questions, the answer should be categorized as not requiring the specific dataset (Answer: NO).

        Please analyze the following question and determine if it is a general question or one that requires information from the specific dataset. 
        Reply with "YES" if it requires the dataset, or "NO" if it does not. Remember your only can reply "YES" or "NO" without reasoning or any other word.

        Question: {question}
        """
        tpl = ChatPromptTemplate.from_template(prompt_template)
        chain = tpl | self.llm
        result = "need rag"
        for i in range(max_retry):
            response = chain.invoke({"question": question})
            if "yes" in response.content.lower():
                break
            elif "no" in response.content.lower():
                result = "no rag"
                break
        return result

    @lru_cache(maxsize=None)
    def decide_rag_or_general(self, question):
        # return self.if_query_rag(question)
        return "need rag"

    def build_chain(self):
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.db_ret, self._history_template()
        )
        question_answer_chain = self.create_custom_stuff_documents_chain(self.llm, self.qa_prompt)

        qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        general_qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.gen_qa_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        general_qa_chain = (
                RunnablePassthrough.assign(answer=general_qa_prompt | self.llm | StrOutputParser())
                | (lambda x: {"answer": x["answer"], "input": x["input"], "chat_history": x["chat_history"],
                              "rag_or_general": "general"})
        )

        def decide_and_cache(inputs):
            question = inputs["input"]
            return self.decide_rag_or_general(question)

        branched_chain = RunnableBranch(
            (lambda x: decide_and_cache(x) == "need rag", qa_chain),
            (lambda x: decide_and_cache(x) == "no rag", general_qa_chain),
            general_qa_chain
        )
        final_chain = (
                RunnablePassthrough().assign(
                    rag_or_general=decide_and_cache
                )
                | branched_chain
        )

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        return RunnableWithMessageHistory(
            final_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def chat(self, query, session_id):
        res = self.rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        return res["answer"], res["rag_or_general"]
