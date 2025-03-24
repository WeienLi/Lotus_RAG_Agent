from functools import lru_cache
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
)
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import queue
import threading
from typing import Dict, Any,List, Optional, Union
from langchain_core.documents import Document
import logging
logging.basicConfig(filename='retrieved_docs.log', level=logging.INFO)
from langchain.schema import BaseRetriever
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


def create_enhanced_history_aware_retriever(
    llm: Any,
    retriever: Any,
    history_prompt: Any,
    context_prompt: Any
) -> Any:
    """Creates an enhanced history-aware retriever that incorporates both past interactions 
    and generated context for improved retrieval performance

    Args:
        llm (Any): The large language model used for generating responses
        retriever (Any): The retriever used for retrieving documents documents
        history_prompt (Any): Prompt template for history-aware retrieval
        context_prompt (Any): prompt template for generating model knowledge context

    Returns:
        Any: A combined retriever chain that merges history-aware retrieval and model-generated context
    """
    
    # Create the history aware retriever chain
    history_aware_retriever = RunnableBranch(
        (
            lambda x: not x.get("chat_history", False),
            (lambda x: x["input"]) | retriever,
        ),
        history_prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="history_aware_retriever_chain")
    
    # Create model internal knowledge context chain
    model_knowledge_chain = (
        context_prompt 
        | llm 
        | StrOutputParser() 
        | (lambda x: [Document(page_content=x, metadata={"source": "model_knowledge"})])
    ).with_config(run_name="model_knowledge_chain")

    #combine chains and the results
    combined_retriever = (
        RunnablePassthrough().assign(
            history_docs=history_aware_retriever,
            model_docs=model_knowledge_chain
        )
        | (lambda x: x["history_docs"] + x["model_docs"])
    ).with_config(run_name="combined_retriever")
    
    return combined_retriever

class ThreadedGenerator:
    """
    A thread-safe generator for streaming data using a queue
    
    Attributes:
        queue(queue): A queue that is used for streaming data.
        
    Methods:
        send(data):
            Adds data to the queue to be streamed

        close():
            Signals the generator to stop by adding a `StopIteration` token
    """
    def __init__(self):
        """
        Intialize the queue
        """
        self.queue = queue.Queue()

    def __iter__(self):
        """Returns the generator itself for iteration

        Returns:
            Threadgenerator: A Threadgenerator for iterations
        """
        return self

    def __next__(self):
        """Retrieves the next item from the queue

        Raises:
            item: Stop Iteration when we try to close it

        Returns:
            obj: The next item in the queue.
        """
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        """Adds data to the queue to be consumed by the generator

        Args:
            data (Any): Data that we want to stream
        """
        self.queue.put(data)

    def close(self):
        """Close the generator
        """
        self.queue.put(StopIteration)



class SelectiveStreamHandler(StreamingStdOutCallbackHandler):
    """
    A callback handler for streaming selected LLM token outputs

    Inherit From:
        Langchain's StreamingStdOutCallbackHandler for handling streaming std out call
    
    Attributes:
        gen(ThreadedGenerator): The generator used to stream tokens
        is_qa_chain(bool): A flag indicating whether it is the final answering chain
        
    Methods:
        on_chain_start(serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
            Checks if the current processing chain is the final answering chain not itermediate context generation or history summrization
        
        on_llm_new_token(token: str, **kwargs):
            Stream only if it is the final qa_chain
    """
    def __init__(self, gen):
        """
        Initializes the selective streaming handler

        Args:
            gen (ThreadedGenerator): The generator used for streaming output
        """
        super().__init__()
        self.gen = gen
        self.is_qa_chain = False

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """
        Checks if the current processing chain is the final answering chain not itermediate context generation or history summrization

        Args:
            serialized (Dict[str, Any]): Serialized metadata about the chain
            inputs (Dict[str, Any]): The input data passed to the chain
        """
        self.is_qa_chain = 'context' in inputs.keys()

    def on_llm_new_token(self, token: str, **kwargs):
        """
        Stream only if it is the final qa_chain

        Args:
            token (str): The new token generated by the LLM.
        """
        if self.is_qa_chain:
            self.gen.send(token)


def format_document(doc):
    """Reformat the documents by appending car_stats at the end

    Args:
        doc (Document): A document object that we want to reformat

    Returns:
        str: A formatted string containing the document content followed by car stats
    """
    page_content = doc.page_content
    car_stats = doc.metadata.get('car_stats', 'No car stats available')
    return f"{page_content}\n Car stats Related: {car_stats}"


class OllamaManager:
    """
    A class use to put together the RAG generation process with Large Language Model from Ollama using Langchain
    
    Attributes:
        llm(ChatOllama) : Large Language Model that will be used for generation
        db_ret (BaseRetriever): The retriever used to fetch relevant documents
        store (Dict[str, ChatMessageHistory]): A dictionary storing chat histories per session
        qa_prompt (str): Prompt template to answer question with RAG needed
        gen_qa_prompt (str): Prompt template if no RAG is needed
        rag_chain (RunnableWithMessageHistory): The full execution chain for answering queries
            
    Methods:
        custom_history_template(contextualize_q_system_prompt):
            Creates a custom history-aware prompt template for reformulating user queries base on history
        
        custom_qa_prompt(prompt):
            Setter methods for changing the final question answering query to prompt
        
        change_model(new_model: str):
            Setter methods for changing the LLM
            
        get_model_info():
            Getter methods for retrieving large language model's info
            
        if_query_rag(question, max_retry=3):
            Determines whether the given questions requires RAG or can be directly answered by LLM.
        
        def decide_rag_or_general(question):
            Cached method to determine whether a query requires RAG
        
        build_chain():
            Constructs the full Retrieval-Augmented Generation (RAG) processing chain.
        
        llm_thread(g, query, session_id):
            Runs the RAG processing pipeline in a real-time streaming thread
            
        chat(self, query, session_id):
            An interface for chatting with our RAG system 
    """
    
    def __init__(self, config, db_ret):
        """Intialize the OllamaManager with configurations and retrivers

        Args:
            config (Dict[str,str]): configurations dictionary for intializing ollama manager 
            db_ret (BaseRetriever): The retriever used to fetch relevant documents

        """
        self.llm = ChatOllama(model=config['llm'])
        self.qa_prompt = self._qa_template()
        self.gen_qa_prompt = self._gen_qa_template()
        self.store = {}
        self.db_ret = db_ret
        self.rag_chain = self.build_chain()
    
    @staticmethod
    def _context_prompt():
        """
        Returns a Langchain's ChatPromptTemplate that use to obtain LLM's internal knowledge
        
        Returns:
            ChatPromptTemplate: Formatted ChatPromptTemplate
        """
        return ChatPromptTemplate.from_messages([
            ("system", """Based on the chat history and your knowledge, provide relevant context 
            or information about the following question. If you don't have specific information, 
            you can provide general context that might be helpful. Your response should be concise 
            and directly related to the question."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    @staticmethod
    def _history_template():
        """
        Returns a Langchain's ChatPromptTemplate for reformulating questions based on chat history.
        
        Returns:
            ChatPromptTemplate: Formatted ChatPromptTemplate
        """
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
        """
        Returns the prompt for final query answering
        
        Returns:
            str: The formatted QA prompt
        """
        return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        Your role is to assist users by answering questions and providing detailed information about Lotus's \
        brand promotion and its famous historical models. Use the following pieces of retrieved context to \
        answer the question.
        Retrieved Context: \n{context} 

        If the user's question is a common, everyday query, such as:
        - "Hello, how are you?"
        - "What's the weather like today?"
        - "How do I make a cup of coffee?"
        - "What's the capital of France?"
        - "What time is it?"
        Respond independently without referring to the retrieved context."""

    @staticmethod
    def _gen_qa_template():
        """
        Returns the prompt for final query answering if no RAG is needed 
        (Not used in current version always go RAG and use qa_template due to performance)
        
        Returns:
            str: The formatted QA prompt
        """
        return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        Your role is to assist users by answering questions and providing detailed information about Lotus's \
        brand promotion and its famous historical models."""

    def custom_history_template(self, contextualize_q_system_prompt):
        """
        Creates a custom history-aware prompt template for reformulating user queries base on history

        Args:
            contextualize_q_system_prompt (str): Prompt that instructs how the query should be reformulated
        """
        self.history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def custom_qa_prompt(self, prompt):
        """Setter methods for changing the final question answering query to prompt

        Args:
            prompt (str): Prompt that we want to set to instruct LLM for final generation
        """
        self.qa_prompt = prompt

    def change_model(self, new_model: str):
        """Setter methods for changing the LLM

        Args:
            new_model (str): Large Language Model that we want to set to change the pipeline
        """
        self.llm = ChatOllama(model=new_model)

    def get_model_info(self):
        """Getter methods for retrieving large language model's info

        Returns:
            Any: Information about the loaded LLM model
        """
        return self.llm.model_info()

    @staticmethod
    def create_custom_stuff_documents_chain(llm, prompt_temp):
        """
        Create a Langchain chain that is used for final generation of RAG by formatting documents 
        as well as taking chat_history and the prompt

        Args:
            llm (ChatOllama): The Large Language Model we want to use for the final answering
            prompt_temp (_type_): The prompt template that will be used to guide the final generation of RAG
        """
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
        """
        Determines whether the given questions requires RAG or can be directly answered by LLM.
        
        Args:
            question(str): The user's query that we want to classify
            max_retry(int, optional): The number of maximum attempts for this classifcation (Ensure no infinite time)
        
        Returns:
            str: "need rag" or "no rag" flag
        """
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
        """
        Cached method to determine whether a query requires RAG
        (On current version always go need rag and include general prompt for final generation due to performance)
        
        Args:
            question (str): The user query

        Returns:
            str: "need rag" or "no rag" 
        """
        # return self.if_query_rag(question)
        return "need rag"
    
    
    def build_chain(self):
        """ 
        Constructs the full Retrieval-Augmented Generation (RAG) processing chain.
        
        Returns:
            RunnableWithMessageHistory: The full LangChain execution chain that processes  queries by retrieving documents 
            genearting reponses and tracking history.
            
        Processing Flow:
        Currently everything is going need RAG so only go left.
        ┌───────────────────────────────────────────────────────────┐
        │                      User Query Input                     │
        └───────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────────────────┐
        │         Determine if query requires retrieval (RAG)       │
        │        (decide_rag_or_general() → "need rag" / "no rag")  │
        └───────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴─────────────────────────┐
        │                                                 │
        ▼                                                 ▼
        ┌────────────────────────────┐             ┌──────────────────────────┐
        │  RAG Needed ("need rag")   │             │ No RAG Needed ("no rag") │
        └────────────────────────────┘             └──────────────────────────┘
                    │                                          │
                    ▼                                          ▼
        ┌────────────────────────────────────┐    ┌──────────────────────────────────┐
        │Retrieve Context (enhanced_retriever)│   │  Use LLM directly (general_qa_chain) │
        └────────────────────────────────────┘    └──────────────────────────────────┘
                    │                                          │
                    ▼                                          ▼
        ┌──────────────────────────────────────────┐   ┌───────────────────────────────┐
        │  Generate Answer (question_answer_chain) │   │ Format response and return    │
        └──────────────────────────────────────────┘   └───────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────────────┐
        │     Return final response to the user       │
        └─────────────────────────────────────────────┘
        """
        enhanced_retriever = create_enhanced_history_aware_retriever(
            self.llm, 
            self.db_ret, 
            self._history_template(),
            self._context_prompt()
        )
        question_answer_chain = self.create_custom_stuff_documents_chain(self.llm, self.qa_prompt)
        qa_chain = create_retrieval_chain(enhanced_retriever, question_answer_chain)
        general_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.gen_qa_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
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
    

        
    def llm_thread(self, g, query, session_id):
        """ Runs the RAG processing pipeline in a real-time streaming thread

        Args:
            g (ThreadedGenerator): The streaming generator
            query (str): The user query
            session_id (str): The session ID for tracking conversation history (Multiple users)
        """
        try:
            selective_handler = SelectiveStreamHandler(g)

            res = self.rag_chain.invoke(
                {"input": query},
                config={
                    "configurable": {"session_id": session_id},
                    "callbacks": [selective_handler]
                },
            )
            # print(res)
            g.send(("FLAG", res.get("rag_or_general", "unknown")))
        finally:
            g.close()

    def chat(self, query, session_id):
        """An interface for chatting with our RAG system 
        (Abstracts the entire code only thing needed to call form API)

        Args:
            query (str): The user query
            session_id (str): The session ID for tracking conversation history

        Returns:
            ThreadedGenerator: A generator streaming responses from the LLM
        """
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, query, session_id)).start()
        return g
