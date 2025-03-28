import os
import sys
import yaml
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chromaManager import ChromaManager
from utils.ollamaManager import OllamaManager


def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def test_retrieval_acc(config, collection_name, test_directory='../Data/test_data', k=10):
    """
    Evaluates the retrieval accuracy of a Chroma database on test queries

    Args:
        config (dict): Configuration dictionary
        collection_name (str): The name of the collection to retrieve data from
        test_directory (str, optional): Directory path containing test queries in JSON files Defaults to '../Data/test_data'
        k (int, optional): The number of top documents we want to retrieve for evaluation Defaults to 10
    """
    manager = ChromaManager(config, collection_name)
    manager.create_collection()

    test_queries = []
    for filename in os.listdir(test_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(test_directory, filename)
            with open(file_path, 'r') as file:
                test_queries.extend(json.load(file))

    accuracy = manager.evaluate_retrieval(test_queries, k, "ensemble", False)
    # accuracy = manager.evaluate_retrieval(test_queries, k, "chroma", False)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def test_if_query_rag(config, collection_name, test_directory):
    """
    Tests how well we can evaluate whether a query should use RAG.

    Args:
        config (dict): Configuration dictionary
        collection_name (str): The name of the collection to retrieve data from
        test_directory (str, optional): Directory path containing test queries in JSON files
    """
    chroma_manager = ChromaManager(config, collection_name)
    chroma_manager.create_collection()
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
    test_retrieval_acc(config, 'lotus_car_stats',
                       '/root/autodl-tmp/RAG_Agent/data/test_data/test_rag', k=10)
    # test_if_query_rag(config, 'lotus', '/root/autodl-tmp/RAG_Agent/Data/test_need_rag')
