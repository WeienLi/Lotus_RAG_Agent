import os
import sys
import json
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apiOllamaManager import ChatManager
from utils.chromaManager import ChromaManager


def get_rag_content(response):
    rag_content = ""
    for i, doc in enumerate(response):
        page_content = doc.page_content.replace('\n', '')
        if len(page_content) < 50:
            continue
        car_stats = doc.metadata.get('car_stats', 'No car stats available')
        car_stat_content = ""
        for car_stat in car_stats:
            car_stat_content += car_stat.replace('\n', ' ')
        rag_content += f"{page_content}\nCar stats Related: {car_stats}\n"
    return rag_content


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def test_chat_manager(config, session_id, retrievers):
    base_url = config['ollama_base_url']
    model_name = config['llm']

    chat_manager = ChatManager(session_id, base_url, model_name)

    print(f"Start to chat with {model_name}.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            break

        # use_rag = chat_manager.if_query_rag(user_input)
        use_rag = 'need rag'
        if use_rag == 'need rag':
            rag_context = ""
            for retriever in retrievers:
                rag_context += get_rag_content(retriever.invoke(user_input))
                rag_context += '\n'
        else:
            rag_context = ""
        response = chat_manager.chat(user_input, rag_context, True)
        ai_response = ""
        print(f"{use_rag} AI response: ", end="")
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    if 'message' in json_line and 'content' in json_line['message']:
                        content = json_line['message']['content']
                        ai_response += content
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass
        print()

        chat_manager.save_chat_history(ai_response)
        print("=" * 50)


if __name__ == "__main__":
    # retriever = None
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    chroma_manager1 = ChromaManager(config=config, collection_name='lotus_car_stats')
    chroma_manager2 = ChromaManager(config=config, collection_name='lotus_brand_info')
    chroma_manager1.create_collection()
    chroma_manager2.create_collection()

    chroma_managers = [chroma_manager1, chroma_manager2]
    retrievers = []
    for chroma_manager in chroma_managers:
        retrievers.append(chroma_manager.get_retriever(k=5, retriever_type="ensemble"))

    test_chat_manager(config, "test_session", retrievers)
