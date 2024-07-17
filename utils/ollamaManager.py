from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class OllamaManager:
    def __init__(self, config):
        self.llm = ChatOllama(model=config['llm'], callbacks=[StreamingStdOutCallbackHandler()])
        # self.llm = Ollama(model=model, callbacks=[StreamingStdOutCallbackHandler()])
        self.template = self._set_template()

    @staticmethod
    def _set_template():
        prompt_template = """
        ## Information:
        {context}

        ## Question:
        {input}
        """
        return ChatPromptTemplate.from_template(prompt_template)

    def chat(self, context: str, question: str):
        chain = self.template | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "input": question})
        return response

    def change_model(self, new_model: str):
        self.llm = ChatOllama(model=new_model, callbacks=[StreamingStdOutCallbackHandler()])

    def set_custom_template(self, custom_template: str):
        self.template = ChatPromptTemplate.from_template(custom_template)

    def get_model_info(self):
        return self.llm.model_info()


if __name__ == "__main__":
    config = {'llm': "llama3"}
    manager = OllamaManager(config)
    context = ""
    while True:
        question = input("Please enter your question: ")
        response = manager.chat(context, question)
        print(response)
