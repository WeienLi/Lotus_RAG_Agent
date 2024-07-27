import os
import sys
import yaml
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.chromaManager import ChromaManager
from src.utils.ollamaManager import OllamaManager


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_db(manager):
    manager.create_collection()
    manager.load_and_store_data()


def remove_collection(manager):
    manager.create_collection()
    manager.remove_all()


if __name__ == '__main__':
    config_path = "../src/config/config.yaml"
    config = load_config(config_path)
    manager = ChromaManager(config, 'lotus')
    # remove_collection(manager)
    load_db(manager)
