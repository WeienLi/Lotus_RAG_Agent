import os
import sys
import yaml
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chromaManager import ChromaManager
from utils.ollamaManager import OllamaManager


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_db(config):
    manager = ChromaManager(config, 'lotus')
    manager.load_model()
    manager.load_and_store_data()


def test_retrieval_acc(config, test_directory='../data/test_data', k=10):
    manager = ChromaManager(config, 'lotus')
    manager.load_model()

    test_queries = []
    for filename in os.listdir(test_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(test_directory, filename)
            with open(file_path, 'r') as file:
                test_queries.extend(json.load(file))

    accuracy = manager.evaluate_retrieval(test_queries, k, "ensemble", False)
    # accuracy = manager.evaluate_retrieval(test_queries, k, "chroma", False)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def test_if_query_rag(config, test_directory):
    chroma_manager = ChromaManager(config, 'lotus')
    chroma_manager.load_model()
    db_ret = chroma_manager.get_db_as_ret(search_kwargs={"k": 10})
    ollama_manager = OllamaManager(config, db_ret)

    test_queries = []
    for filename in os.listdir(test_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(test_directory, filename)
            with open(file_path, 'r') as file:
                test_queries.extend(json.load(file))

    num = 0
    for test_query in tqdm(test_queries):
        question = test_query['question']
        true_label = test_query['label']
        test_label = ollama_manager.if_query_rag(question)
        if true_label == test_label:
            num += 1
        else:
            print(f"Question: {question}\ntrue_label: {true_label}, test_label: {test_label}")

    print(f"Total test query: {len(test_queries)}, "
          f"Correct Number: {num}, "
          f"Accuracy: {num / len(test_queries) * 100:.2f}%")


if __name__ == '__main__':
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    # test_retrieval_acc(config, '/root/autodl-tmp/RAG_Agent/data/test_data', k=10)
    test_if_query_rag(config, '/root/autodl-tmp/RAG_Agent/data/test_need_rag')
