import requests
import json
from sseclient import SSEClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


class ChatManager:
    """
    Manages the conversational interactions without the used of Langchain for LLM-based Retrieval-Augmented Generation (RAG) system
    
    Attributes:
        session_id (str): Identifier for the chat session
        base_url (str): Base URL of the external LLM API
        model_name (str): Name of the LLM model being used
        llm (ChatOllama): Chat model instance for processing queries
        chat_history (list[dict]): Limited history of user-assistant messages for context
        all_chat_history (list[dict]): Complete chat history, including all messages
        history_limit (int): Maximum number of messages retained in chat history

    Methods:
        if_query_rag(question, max_retry=3):
            Determines whether the given questions requires RAG or can be directly answered by LLM

        chat(user_input, rag_context='', stream=False):
            An interface for chatting with our RAG system
        
        save_chat_history(response):
            Saves the assistant's response to chat history
        
        _trim_chat_history():
            Trims chat history to maintain only the last `history_limit` messages while not popping system message.
        
        def get_chat_history():
            Getter methods for retrieving chat history
        
        def get_all_chat_history():
            Getter methods for retrieving the entire chat history
        
        clear_chat_history():
            Clears the current chat history while retaining the system message
    """
    def __init__(self, session_id, base_url, model_name, history_limit=10):
        """
        Initializes the ChatManager with configuration

        Args:
            session_id (str): Session identifier
            base_url (str): API base URL for sending chat requests
            model_name (str): The LLM model used for answering queries
            history_limit (int, optional): Maximum chat messages retained (must be even) Defaults to 10
        """
        assert history_limit % 2 == 0, "history_limit must be an even number" #One user one model
        self.session_id = session_id
        self.base_url = base_url
        self.model_name = model_name
        self.llm = ChatOllama(model=self.model_name)
        self.chat_history = [{
            "role": "system", "content": self._sys_template()
        }]
        self.all_chat_history = [{
            "role": "system", "content": self._sys_template()
        }]
        self.history_limit = history_limit

    @staticmethod
    def _sys_template():
        """
        Returns the system prompt
        
        Returns:
            str: System prompt
        """
        return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        Your role is to assist users by answering questions and providing detailed information about Lotus's \
        brand promotion and its famous historical models. You answers questions \
        based on snippets of text provided in context. Answer should being as concise as possible."""

    @staticmethod
    def _qa_template(question, context):
        """
        Formats the user question into a structured prompt depending on whether context is required:
        
        Args:
            question (str): The user's question
            context (str): retrived context from RAG
        
        Returns:
            str: formatted query prompt
        """
        if context:
            return f"""You can respond to the user input based on the following Retrieved Context.
            Retrieved Context: \n{context}\n
            User input: {question}"""
        else:
            return f"""{question}"""

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

    def chat(self, user_input, rag_context='', stream=False):
        """An interface for chatting with our RAG system 
        (Abstracts the entire code only thing needed to call form API)

        Args:
            user_input (str): The user query
            rag_context (str, optional): retrieved contexts for RAG Defaults to ''
            stream (bool, optional): Boolean on whether we want to stream the ouput Defaults to False

        Returns:
            requests.Response: The HTTP response containing the assistant's reply
        """
        user_message = {"role": "user", "content": self._qa_template(user_input, rag_context)}
        self.chat_history.append(user_message)
        self.all_chat_history.append(user_message)
        print(len(self.chat_history), self.chat_history)

        data = {
            "model": self.model_name,
            "messages": self.chat_history,
            "stream": stream
        }

        response = requests.post(f"{self.base_url}/api/chat", json=data, stream=True)
        response.raise_for_status()

        return response

    def save_chat_history(self, response):
        """
         Saves the assistant's response to chat history

        Args:
            response (str): The generated response from the LLM.
        """
        assistant_message = {"role": "assistant", "content": response}
        self.chat_history.append(assistant_message)
        self.all_chat_history.append(assistant_message)
        self._trim_chat_history()

    def _trim_chat_history(self):
        """
        Trims chat history to maintain only the last `history_limit` messages while not popping system message.
        """
        # Keep the system message and the last `self.history_limit` user and assistant messages
        non_system_messages = [msg for msg in self.chat_history if msg['role'] != 'system']
        if len(non_system_messages) > self.history_limit:
            self.chat_history = [self.chat_history[0]] + non_system_messages[-self.history_limit:]

    def get_chat_history(self):
        """
        Getter methods for retrieving chat history

        Returns:
            list[dict]: List of chat history
        """
        return self.chat_history

    def get_all_chat_history(self):
        """
        Getter methods for retrieving the entire chat history

        Returns:
            list[dict]: Full list of messages from the chat session
        """
        return self.all_chat_history

    def clear_chat_history(self):
        """
        Clears the current chat history while retaining the system message
        """
        self.chat_history = [self.all_chat_history[0]]
