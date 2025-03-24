import os
import sys
import json
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apiOllamaManager import ChatManager
from utils.chromaManager import ChromaManager


def get_rag_content(response):
    """
    Extracts content from documents retrieved and reformat it

    Args:
        response (list): A list of document objects

    Returns:
        str: A formatted string containing extracted text follow by car stats.
    """
    rag_content = ""
    for i, doc in enumerate(response):
        page_content = doc.page_content.replace('\n', '')
        if len(page_content) < 50:
            continue
        car_stats = doc.metadata.get('car_stats', None)
        if car_stats:
            car_stat_content = ""
            for car_stat in car_stats:
                car_stat_content += car_stat.replace('\n', ' ')
            rag_content += f"{page_content}\nCar stats Related: {car_stat_content}\n"
        else:
            rag_content += f"{page_content}\n"

    return rag_content


def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def test_chat_manager(config, session_id, retrievers, questions):
    """
    Test the ChatManager of apiOllamaManager for its RAG generation using questions.

    Args:
        config (dict): Configuration dictionary
        session_id (str): Session identifier for the chat session
        retrievers (list[base_retriever]): A list of retriever objects used for RAG-based query retrieval
        questions (list[str]): A list of input queries to be tested

    Returns:
        list[dict]: A list of dictionaries containing the input query and answer pair
    """
    base_url = config['ollama_base_url']
    model_name = config['llm']

    chat_manager = ChatManager(session_id, base_url, model_name)

    print(f"Start to chat with {model_name}.")

    results = []

    for user_input in questions:
        print(f"You: {user_input}")

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

        results.append({
            "question": user_input,
            "response": ai_response
        })

    return results


if __name__ == "__main__":
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}
    retrievers = []
    for collection, top_k in collections.items():
        if top_k <= 0:
            continue
        chroma_manager = ChromaManager(config=config, collection_name=collection)
        chroma_manager.create_collection()
        retrievers.append(chroma_manager.get_retriever(k=top_k, retriever_type="ensemble"))

 
    questions = [
        "Sensitive INFO"
    ]

    results = test_chat_manager(config, "test_session", retrievers, questions)

    with open('/root/autodl-tmp/RAG_Agent/data/test_data/q1.txt', 'w') as outfile:
        for result in results:
            outfile.write(f"Question: {result['question']}\n")
            outfile.write(f"Response: {result['response']}\n")
            outfile.write("=" * 50 + "\n")
