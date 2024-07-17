from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
)
def format_document(doc):
    page_content = doc.page_content
    car_stats = doc.metadata.get('car_stats', 'No car stats available')
    return f"{page_content}\n Car stats Related: {car_stats}"

class OllamaManager:
    def __init__(self, config,db_ret):
        self.llm = ChatOllama(model=config['llm'])
        self.history_prompt = self._history_template()
        self.qa_prompt = """You are an assistant for question-answering tasks. \
                Use the following pieces of retrieved context to answer the question. \
                If you don't know the answer, just say that you don't know. \
                Use three sentences maximum and keep the answer concise.\
                {context}"""
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
    
    def custom_history_template(self,contextualize_q_system_prompt):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.history_prompt = contextualize_q_prompt

    # need to rebuild the chain
    def custom_qa_prompt(self,prompt):
        self.qa_prompt = prompt
    # need to rebuild the chain
    def change_model(self, new_model: str):
        self.llm = ChatOllama(model=new_model)

    def get_model_info(self):
        return self.llm.model_info()

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

    def build_chain(self):
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.db_ret, self.history_prompt
        )
        question_answer_chain = self.create_custom_stuff_documents_chain(self.llm, self.qa_prompt)
        
        qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        return RunnableWithMessageHistory(
            qa_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    def chat(self,query,session_id):
        #print(self.rag_chain)
        res = self.rag_chain.invoke({"input": query},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        #print(res)
        return res["answer"]