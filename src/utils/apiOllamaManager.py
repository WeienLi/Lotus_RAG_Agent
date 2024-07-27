import requests
import json
from sseclient import SSEClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


class ChatManager:
    def __init__(self, session_id, base_url, model_name):
        self.session_id = session_id
        self.base_url = base_url
        self.model_name = model_name
        self.llm = ChatOllama(model=self.model_name)
        self.chat_history = []
        self.chat_history.append({
            "role": "system", "content": self._sys_template()
        })

    @staticmethod
    def _sys_template():
        return """You are Colin, an LLM-driven guide for Lotus Starlight Avenue. \
        Your role is to assist users by answering questions and providing detailed information about Lotus's \
        brand promotion and its famous historical models. You answers questions \
        based on snippets of text provided in context. Answer should being as concise as possible."""

    @staticmethod
    def _qa_template(question, context):
        return f"""Retrieved content: \n{context}\nBased on the above retrieved content to \
        answer the question if you find the content is useful. if you think the following \
        question is not related to the content, just directly answer the question on your own
        Question: \n{question}"""

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

    def chat(self, user_input, rag_context='', stream=False):
        self.chat_history.append({"role": "user", "content": self._qa_template(user_input, rag_context)})

        data = {
            "model": self.model_name,
            "messages": self.chat_history,
            "stream": stream
        }

        response = requests.post(f"{self.base_url}/api/chat", json=data, stream=True)
        response.raise_for_status()

        return response

    def save_chat_history(self, response):
        self.chat_history.append({"role": "assistant", "content": response})
        # print(len(self.chat_history))

    def get_chat_history(self):
        return self.chat_history

    def clear_chat_history(self):
        self.chat_history = []
